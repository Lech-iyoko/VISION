import time
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL, EOT_BACKEND, EOT_THRESHOLD,
    EOT_CHECK_SILENCE_MS, EOT_RECHECK_SILENCE_MS, EOT_MAX_SILENCE_MS,
    SPECULATIVE_ASR_ENABLED,
)

torch.set_num_threads(1)


class LocalVoiceStreamer:
    """
    Drop-in replacement for VoiceStreamer using fully local ASR.

    Pipeline:
      Microphone → Silero VAD (segmentation + barge-in) → faster-whisper → on_final_transcript

    Silero VAD is used for two separate jobs:
      1. Speech segmentation: detect when the user starts/stops speaking → triggers transcription
      2. Barge-in detection: detect speech during TTS playback → triggers on_barge_in

    End-of-turn detection (EOT_BACKEND = "smart_turn"):
      Instead of a fixed silence window, smart-turn v3 scores the waveform at
      silence checkpoints (160ms, 576ms). A confident "turn complete" ends the
      turn immediately; "not done" extends the wait up to EOT_MAX_SILENCE_MS so
      mid-thought pauses aren't cut off. The check runs on its own CPU thread —
      the audio thread never blocks; it just reads a completion flag.

    Speculative ASR (SPECULATIVE_ASR_ENABLED):
      Whisper starts transcribing the buffered speech at the first silence
      checkpoint, in parallel with the turn decision. If the user resumes
      speaking, the episode counter invalidates the result; if the turn ends,
      the transcript is already done → Whisper leaves the critical path.

    Same public interface as VoiceStreamer so orchestrator needs no changes.
    """

    SAMPLE_RATE = 16000
    VAD_CHUNK = 512           # samples per VAD frame (32ms at 16kHz)
    BLOCK_SIZE = 1024         # sounddevice callback block (64ms)

    def __init__(
        self,
        on_final_transcript: Callable[[str], None],
        speech_threshold: float = 0.5,
        barge_in_threshold: float = 0.7,
        silence_frames_to_end: int = 15,   # 15 × 32ms = 480ms silence → end of turn
        min_speech_frames: int = 8,         # 8 × 32ms = 256ms minimum to bother transcribing
    ):
        self.on_final_transcript = on_final_transcript
        self.on_barge_in: Optional[Callable[[], None]] = None

        # ── Mute / cooldown ───────────────────────────────────────────────
        self.is_muted = False
        self.unmute_time = 0.0
        self.cooldown_seconds = 1.5

        # ── Barge-in state ────────────────────────────────────────────────
        self.tts_is_playing = False
        self.barge_in_triggered = False
        self._barge_in_lock = threading.Lock()
        self.barge_in_threshold = barge_in_threshold
        self._barge_consecutive = 0
        self._barge_required_frames = 5
        self._barge_speaking = False
        self._barge_speech_start = 0.0
        self._barge_min_duration = 0.5

        # ── Speech segmentation state ─────────────────────────────────────
        self.speech_threshold = speech_threshold
        self._silence_limit = silence_frames_to_end
        self._min_speech = min_speech_frames
        self._speech_active = False
        self._speech_buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0

        # ── End-of-turn detection (smart-turn v3) ─────────────────────────
        # All buffer mutations happen on the audio thread. The smart-turn
        # worker only reads an audio snapshot and writes _eot_complete_episode;
        # the audio thread finalizes on its next frame. If the user resumes
        # speaking first, the episode counter advances and the stale flag
        # can never match.
        frame_ms = self.VAD_CHUNK * 1000 // self.SAMPLE_RATE  # 32ms
        self._eot_check_frames = (
            EOT_CHECK_SILENCE_MS // frame_ms,
            EOT_RECHECK_SILENCE_MS // frame_ms,
        )
        self._eot_max_frames = EOT_MAX_SILENCE_MS // frame_ms
        self._silence_episode = 0        # advances when speech resumes or segment resets
        self._eot_complete_episode = -1  # episode the detector marked complete
        self._last_eot_prob = -1.0
        self._eot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart-turn")

        # ── Speculative ASR ───────────────────────────────────────────────
        self._spec_enabled = SPECULATIVE_ASR_ENABLED
        self._spec_future = None         # in-flight Whisper job for current episode

        # Per-turn timing for the eval logger (read by orchestrator)
        self.last_turn_timing: dict = {}

        # ── VAD input buffer ──────────────────────────────────────────────
        self._vad_buf = np.array([], dtype=np.float32)
        self._vad_lock = threading.Lock()

        # ── Background transcription ──────────────────────────────────────
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._stop = threading.Event()

        self._setup_vad()
        self._setup_whisper()
        self._setup_turn_detector()

    # ── model loading ─────────────────────────────────────────────────────

    def _setup_vad(self):
        print("Loading Silero VAD model...")
        try:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad(onnx=True)
            print("✅ Silero VAD loaded (ONNX mode)")
        except ImportError:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            print("✅ Silero VAD loaded via torch.hub")

    def _setup_whisper(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for compute_type in (["float16", "int8"] if device == "cuda" else ["int8"]):
            try:
                print(f"Loading Whisper model ({WHISPER_MODEL}) on {device.upper()} [{compute_type}]...")
                self.whisper = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
                print(f"✅ faster-whisper loaded ({WHISPER_MODEL} | {device.upper()} | {compute_type})")
                # Pre-warm CUDA context — eliminates 7s cold start on first real turn
                print("   Warming up GPU...")
                silence = np.zeros(self.SAMPLE_RATE, dtype=np.float32)
                list(self.whisper.transcribe(silence, language="en")[0])
                print("   ✅ GPU warm")
                return
            except Exception as e:
                print(f"   [{compute_type}] failed: {e} — trying next...")
        raise RuntimeError("faster-whisper: no supported compute type available")

    def _setup_turn_detector(self):
        """Load smart-turn v3, falling back to fixed-silence EOT on any failure."""
        self.turn_detector = None
        if EOT_BACKEND != "smart_turn":
            print(f"ℹ️  End-of-turn: fixed silence ({self._silence_limit * 32}ms)")
            return
        try:
            from services.voice.turn_detector import TurnDetector
            self.turn_detector = TurnDetector(threshold=EOT_THRESHOLD)
        except Exception as e:
            print(f"⚠️  smart-turn unavailable ({e}) — falling back to fixed silence EOT")

    # ── audio processing ──────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[MIC] {status}", end="\r")
        # sounddevice gives float32 in [-1, 1]; convert to int16 bytes for compat
        audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        self._process_audio(audio_bytes)

    def _process_audio(self, audio_bytes: bytes):
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self._vad_lock:
            self._vad_buf = np.concatenate([self._vad_buf, audio_f32])

        while True:
            with self._vad_lock:
                if len(self._vad_buf) < self.VAD_CHUNK:
                    break
                chunk = self._vad_buf[:self.VAD_CHUNK].copy()
                self._vad_buf = self._vad_buf[self.VAD_CHUNK:]

            prob = self.vad_model(torch.from_numpy(chunk), self.SAMPLE_RATE).item()

            # barge-in (only during TTS playback)
            if self.tts_is_playing and not self.barge_in_triggered:
                if prob > self.barge_in_threshold:
                    self._on_barge_speech()
                else:
                    self._on_barge_silence()

            # speech segmentation (only when not playing TTS)
            if not self.tts_is_playing:
                self._segment(chunk, prob > self.speech_threshold)

    def _segment(self, chunk: np.ndarray, is_speech: bool):
        if is_speech:
            if self._silence_frames > 0:
                # Speech resumed mid-pause: this silence episode is over.
                # Any in-flight smart-turn result or speculative transcript
                # for it is now stale.
                self._silence_episode += 1
                self._spec_future = None
            self._speech_active = True
            self._speech_frames += 1
            self._silence_frames = 0
            self._speech_buffer.append(chunk)
            return

        if not self._speech_active:
            return

        self._silence_frames += 1
        self._speech_buffer.append(chunk)  # keep trailing silence for naturalness

        # At silence checkpoints: score the turn (CPU thread) and start
        # transcribing speculatively (GPU thread). Neither blocks this thread.
        if self.turn_detector and self._silence_frames in self._eot_check_frames:
            audio = np.concatenate(self._speech_buffer)
            self._eot_executor.submit(self._check_turn, self._silence_episode, audio)
            if (self._spec_enabled and self._spec_future is None
                    and self._speech_frames >= self._min_speech):
                self._spec_future = self._executor.submit(self._spec_transcribe, audio)

        if self.turn_detector:
            turn_complete = self._eot_complete_episode == self._silence_episode
            hard_cap = self._silence_frames >= self._eot_max_frames
        else:
            turn_complete = False
            hard_cap = self._silence_frames >= self._silence_limit

        if turn_complete or hard_cap:
            self._finalize_segment()

    def _check_turn(self, episode: int, audio: np.ndarray):
        """Smart-turn worker: score the segment, flag completion for the audio thread."""
        t0 = time.time()
        prob = self.turn_detector.predict(audio)
        ms = int((time.time() - t0) * 1000)
        self._last_eot_prob = prob
        verdict = "complete" if prob >= self.turn_detector.threshold else "not done"
        print(f"🎯 smart-turn: P={prob:.2f} → {verdict} [{ms}ms]")
        if prob >= self.turn_detector.threshold:
            self._eot_complete_episode = episode

    def _finalize_segment(self):
        """End the turn: hand audio (or the speculative transcript) to Whisper."""
        if self._speech_frames >= self._min_speech:
            timing = {
                "eot_silence_ms": self._silence_frames * 32,
                "eot_prob": round(self._last_eot_prob, 3),
                "asr_speculative": self._spec_future is not None,
            }
            if self._spec_future is not None:
                # Transcript already in flight — just deliver its result.
                # Same single-worker executor as the spec job, so this queues
                # behind it and never deadlocks.
                self._executor.submit(self._deliver_speculative, self._spec_future, timing)
            else:
                audio = np.concatenate(self._speech_buffer)
                self._executor.submit(self._transcribe, audio, timing)
        self._reset_segment()

    def _reset_segment(self):
        self._speech_active = False
        self._speech_buffer = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._silence_episode += 1   # invalidate any in-flight EOT/spec results
        self._spec_future = None
        self._last_eot_prob = -1.0

    # ── barge-in ──────────────────────────────────────────────────────────

    def _on_barge_speech(self):
        with self._barge_in_lock:
            self._barge_consecutive += 1
            if self._barge_consecutive < self._barge_required_frames:
                return
            if not self._barge_speaking:
                self._barge_speaking = True
                self._barge_speech_start = time.time()
                print("🎤 VAD: Sustained speech detected...")
            else:
                if time.time() - self._barge_speech_start >= self._barge_min_duration:
                    print(f"🛑 BARGE-IN triggered! (speech: {time.time() - self._barge_speech_start:.2f}s)")
                    self.barge_in_triggered = True
                    if self.on_barge_in:
                        self.on_barge_in()

    def _on_barge_silence(self):
        with self._barge_in_lock:
            self._barge_consecutive = 0
            self._barge_speaking = False

    # ── transcription ─────────────────────────────────────────────────────

    def _run_whisper(self, audio: np.ndarray) -> tuple[str, int]:
        """Whisper inference only. Returns (text, latency_ms)."""
        t0 = time.time()
        segments, _ = self.whisper.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, int((time.time() - t0) * 1000)

    def _transcribe(self, audio: np.ndarray, timing: dict | None = None):
        """Normal path: transcribe a finalized segment, then deliver."""
        try:
            text, latency_ms = self._run_whisper(audio)
            self._deliver(text, latency_ms, timing)
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")

    def _spec_transcribe(self, audio: np.ndarray) -> tuple[str, int]:
        """Speculative path: transcribe before the turn has ended. Returns
        (text, latency_ms) via the future; delivery happens only if the
        segment finalizes while this result is still valid."""
        return self._run_whisper(audio)

    def _deliver_speculative(self, future, timing: dict):
        """Deliver a speculative transcript once the segment has finalized."""
        try:
            text, latency_ms = future.result()
            self._deliver(text, latency_ms, timing)
        except Exception as e:
            print(f"❌ Speculative transcription error: {e}")

    def _deliver(self, text: str, latency_ms: int, timing: dict | None = None):
        """Shared gate: mute/cooldown checks, then the transcript callback."""
        if not text:
            return

        spec = " (speculative)" if timing and timing.get("asr_speculative") else ""
        print(f"\n📝 FINAL: {text:<80s}  [{latency_ms}ms{spec}]")

        if self.is_muted:
            print("   ↳ (ignored - muted)")
            return

        time_since_unmute = time.time() - self.unmute_time
        if time_since_unmute < self.cooldown_seconds:
            print(f"   ↳ (ignored - cooldown: {time_since_unmute:.1f}s < {self.cooldown_seconds}s)")
            return

        if timing is not None:
            timing["asr_ms"] = latency_ms
            self.last_turn_timing = timing

        self.on_final_transcript(text)

    # ── public interface (matches VoiceStreamer exactly) ──────────────────

    def mute(self):
        self.is_muted = True
        self._reset_segment()
        with self._vad_lock:
            self._vad_buf = np.array([], dtype=np.float32)
        if hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()
        print("🔇 Transcript processing MUTED")

    def unmute(self):
        self.is_muted = False
        self.unmute_time = time.time()
        self._barge_speaking = False
        self._barge_consecutive = 0
        self._reset_segment()
        with self._vad_lock:
            self._vad_buf = np.array([], dtype=np.float32)
        if hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()
        print(f"🎤 Transcript processing UNMUTED (cooldown: {self.cooldown_seconds}s)")

    def start_tts_playback(self):
        self.tts_is_playing = True
        self.barge_in_triggered = False
        self._barge_speaking = False
        self._barge_consecutive = 0
        with self._vad_lock:
            self._vad_buf = np.array([], dtype=np.float32)
        if hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()
        print("🔊 TTS playback started - barge-in detection active")

    def stop_tts_playback(self):
        self.tts_is_playing = False
        self.barge_in_triggered = False
        self._barge_speaking = False
        self._barge_consecutive = 0
        print("🔇 TTS playback stopped - barge-in detection paused")

    def start_streaming(self):
        print("🎤 Opening microphone (local ASR — faster-whisper)...")
        print(f"   Sample rate: {self.SAMPLE_RATE} Hz")
        print("🎤 Speak naturally. Press Ctrl+C to stop.\n")

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=self.BLOCK_SIZE,
                callback=self._audio_callback,
            ):
                print("✅ Microphone open. Listening...")
                self._stop.wait()
        except KeyboardInterrupt:
            print("\n⏸️ Stream interrupted by user")
        finally:
            self._stop.set()
            self._executor.shutdown(wait=False)
            self._eot_executor.shutdown(wait=False)
            print("✅ Local ASR stream stopped.")


if __name__ == "__main__":
    def on_transcript(text: str):
        print(f"\n🎯 Transcript: {text}\n")

    streamer = LocalVoiceStreamer(on_final_transcript=on_transcript)
    streamer.start_streaming()
