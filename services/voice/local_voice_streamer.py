import time
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from faster_whisper import WhisperModel

from config import WHISPER_MODEL

torch.set_num_threads(1)


class LocalVoiceStreamer:
    """
    Drop-in replacement for VoiceStreamer using fully local ASR.

    Pipeline:
      Microphone → Silero VAD (segmentation + barge-in) → faster-whisper → on_final_transcript

    Silero VAD is used for two separate jobs:
      1. Speech segmentation: detect when the user starts/stops speaking → triggers transcription
      2. Barge-in detection: detect speech during TTS playback → triggers on_barge_in

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

        # ── VAD input buffer ──────────────────────────────────────────────
        self._vad_buf = np.array([], dtype=np.float32)
        self._vad_lock = threading.Lock()

        # ── Background transcription ──────────────────────────────────────
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._stop = threading.Event()

        self._setup_vad()
        self._setup_whisper()

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
        print("✅ LocalVoiceStreamer initialized")
        print(f"   Speech threshold: {self.speech_threshold} | Barge-in threshold: {self.barge_in_threshold}")
        print(f"   End-of-turn silence: {self._silence_limit * 32}ms")

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
            self._speech_active = True
            self._speech_frames += 1
            self._silence_frames = 0
            self._speech_buffer.append(chunk)
        else:
            if self._speech_active:
                self._silence_frames += 1
                self._speech_buffer.append(chunk)  # keep trailing silence for naturalness

                if self._silence_frames >= self._silence_limit:
                    if self._speech_frames >= self._min_speech:
                        audio = np.concatenate(self._speech_buffer)
                        self._executor.submit(self._transcribe, audio)
                    self._reset_segment()

    def _reset_segment(self):
        self._speech_active = False
        self._speech_buffer = []
        self._speech_frames = 0
        self._silence_frames = 0

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

    def _transcribe(self, audio: np.ndarray):
        t0 = time.time()
        try:
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
            latency_ms = int((time.time() - t0) * 1000)

            if not text:
                return

            print(f"\n📝 FINAL: {text:<80s}  [{latency_ms}ms]")

            if self.is_muted:
                print("   ↳ (ignored - muted)")
                return

            time_since_unmute = time.time() - self.unmute_time
            if time_since_unmute < self.cooldown_seconds:
                print(f"   ↳ (ignored - cooldown: {time_since_unmute:.1f}s < {self.cooldown_seconds}s)")
                return

            self.on_final_transcript(text)
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")

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
            print("✅ Local ASR stream stopped.")


if __name__ == "__main__":
    def on_transcript(text: str):
        print(f"\n🎯 Transcript: {text}\n")

    streamer = LocalVoiceStreamer(on_final_transcript=on_transcript)
    streamer.start_streaming()
