"""
Orchestrator - The central coordinator for VISION.

Audio pipeline:
  VoiceStreamer (ASR) → AudioFrontend (AEC + VAD) → OllamaMultimodalClient (LLM+vision) → ElevenLabsClient (TTS)

Vision:
  ScreenStateTracker  — background thread, captures screen every 5s, sends changed
                        frames to Gemini 2.5 Flash, maintains temporal workspace context
  FrameCapture        — captures one fresh camera frame at turn time (physical world)

Proactive trigger:
  When the screen tracker detects a high-signal event (error, crash, traceback)
  and the system is idle-listening, VISION speaks up unprompted. Gated by a
  cooldown and a similarity check so the same issue is only announced once.

Fusion:
  FusionEngine combines transcript + screen workspace state into a grounded prompt.
  Camera frame is passed separately as a visual attachment to the multimodal LLM.

Conversation state machine:
  IDLE → LISTENING → THINKING → SPEAKING → IDLE

Latency strategy: stream LLM tokens, buffer into sentences, TTS each sentence while
the next one is still being generated. Audio begins after the first sentence (~500ms
of LLM output) rather than after the full response.
"""

import re
import time
import threading
from difflib import SequenceMatcher
import sounddevice as sd
from services.voice.voice_streamer import VoiceStreamer
from services.voice.local_voice_streamer import LocalVoiceStreamer
from services.voice.audio_frontend import AudioFrontend, AudioConfig
from services.gemini_llm_client import GeminiLLMClient
from services.voice.tts_client import ElevenLabsClient
from services.voice.kokoro_tts_client import KokoroTTSClient
from services.voice.echo_guard import is_echo
from services.vision.frame_capture import FrameCapture
from services.vision.screen_state import ScreenStateTracker
from services.vision.fusion_engine import FusionEngine

from dotenv import load_dotenv
from config import (
    DEFAULT_VOICE_ID, SYSTEM_PROMPT, GEMINI_LLM_MODEL, TTS_BACKEND, ASR_BACKEND,
    PROACTIVE_ENABLED, PROACTIVE_COOLDOWN_S, PROACTIVE_SIMILARITY_THRESHOLD,
)
from utils.eval_logger import EvaluationLogger

load_dotenv()

# Split on sentence-ending punctuation followed by whitespace.
# Won't split inside numbers (3.14) or most abbreviations (no trailing space).
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


class ConversationState:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self):
        print("Initializing VISION components...")

        # === Core Components ===
        self.llm_client = GeminiLLMClient(system_prompt=SYSTEM_PROMPT)
        if TTS_BACKEND == "kokoro":
            self.tts_client = KokoroTTSClient()
        else:
            self.tts_client = ElevenLabsClient(voice_id=DEFAULT_VOICE_ID)

        # Audio processing pipeline
        self.audio_frontend = AudioFrontend(AudioConfig(
            sample_rate=16000,
            vad_threshold=0.5,
            onset_frames=3,
            silence_frames=15,
            aec_enabled=True
        ))

        # ASR streaming
        if ASR_BACKEND == "local":
            self.voice_streamer = LocalVoiceStreamer(
                on_final_transcript=self._handle_transcript
            )
        else:
            self.voice_streamer = VoiceStreamer(
                on_final_transcript=self._handle_transcript
            )

        # === Wire up callbacks ===
        self.audio_frontend.on_barge_in = self._handle_barge_in
        self.voice_streamer.on_barge_in = self._handle_barge_in

        # === State Management ===
        self.state = ConversationState.IDLE
        self.state_lock = threading.Lock()

        # Echo detection
        self.last_tts_text = ""

        # Barge-in control
        self._barge_in_requested = False
        self._barge_in_lock = threading.Lock()
        self._turn_barged_in = False

        # === Vision ===
        self.framecapture = FrameCapture(camera_index=0)
        self.screen_tracker = ScreenStateTracker(
            on_proactive_trigger=self._handle_screen_alert
        )
        self.fusion = FusionEngine()

        # Proactive alert dedup — remember what was last announced and when
        self._last_proactive_desc = ""
        self._last_proactive_time = 0.0

        # Evaluation logger
        self.eval_logger = EvaluationLogger(
            llm_model=GEMINI_LLM_MODEL,
            vlm_model="gemini-2.5-flash",
        )

    # ── state ────────────────────────────────────────────────────────────────

    def _set_state(self, new_state: str):
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            print(f"📍 State: {old_state} → {new_state}")

    # ── barge-in ─────────────────────────────────────────────────────────────

    def _handle_barge_in(self):
        with self._barge_in_lock:
            if self.state == ConversationState.SPEAKING:
                self._barge_in_requested = True
                self._turn_barged_in = True
                print("🛑 Barge-in requested!")

    def _check_barge_in(self) -> bool:
        with self._barge_in_lock:
            if self._barge_in_requested:
                self._barge_in_requested = False
                return True
            return False

    # ── proactive trigger ────────────────────────────────────────────────────

    def _handle_screen_alert(self, description: str):
        """
        Called by ScreenStateTracker (on its background thread) when a
        high-signal screen event is detected. Decides whether to interject,
        then hands off to a turn thread — this method must return quickly
        so the screen loop is never blocked by speech.
        """
        if not PROACTIVE_ENABLED:
            return

        if time.time() - self._last_proactive_time < PROACTIVE_COOLDOWN_S:
            print("🔕 Proactive alert suppressed (cooldown)")
            return

        similarity = SequenceMatcher(
            None, description.lower(), self._last_proactive_desc.lower()
        ).ratio()
        if similarity >= PROACTIVE_SIMILARITY_THRESHOLD:
            print(f"🔕 Proactive alert suppressed (same issue, {similarity:.0%} match)")
            return

        # Claim the turn atomically: only interject when idle-listening,
        # and flip to THINKING inside the lock so a user transcript arriving
        # at the same moment can't start a competing turn.
        with self.state_lock:
            if self.state != ConversationState.LISTENING:
                print(f"🔕 Proactive alert dropped (state={self.state})")
                return
            self.state = ConversationState.THINKING
            print(f"📍 State: {ConversationState.LISTENING} → {self.state} (proactive)")

        self._last_proactive_time = time.time()
        self._last_proactive_desc = description

        threading.Thread(
            target=self._proactive_turn, args=(description,),
            daemon=True, name="proactive-turn",
        ).start()

    def _proactive_turn(self, description: str):
        """
        Speak an unprompted alert about a screen event. Mirrors the
        _handle_transcript flow but with a synthetic prompt and no camera
        frame — the screen event itself is the context.
        """
        print(f"\n🚨 [Proactive]: {description[:100]}")

        turn_start = time.time()
        self._turn_barged_in = False
        self.voice_streamer.mute()

        try:
            prompt = (
                "[PROACTIVE ALERT — you just noticed this on the user's screen. "
                "He has not said anything. Briefly alert him in one or two "
                "spoken sentences: what you saw and, if obvious, a next step. "
                "Do not mention that this is an automated alert.]\n\n"
                f"{description}"
            )

            self._set_state(ConversationState.SPEAKING)

            llm_start = time.time()
            full_response, first_audio_ms = self._stream_response_to_speech(prompt, None)
            llm_ms = int((time.time() - llm_start) * 1000)

            print(f"[VISION, unprompted]: {full_response}")

            self.last_tts_text = full_response
            e2e_ms = int((time.time() - turn_start) * 1000)

            self.eval_logger.log_turn(
                transcript="[PROACTIVE]",
                response=full_response,
                visual_context_preview=description[:150],
                vision_age_before_s=0.0,
                vision_fetch_ms=0,
                llm_latency_ms=llm_ms,
                e2e_to_speech_ms=e2e_ms,
                barge_in=self._turn_barged_in,
                time_to_first_audio_ms=first_audio_ms,
            )

        finally:
            self._set_state(ConversationState.LISTENING)
            time.sleep(0.3)
            self.voice_streamer.unmute()
            self.audio_frontend.reset()

        print("\n✅ Ready for next input...")

    # ── main conversation loop ────────────────────────────────────────────────

    def _handle_transcript(self, transcript: str):
        if not transcript.strip():
            return

        if is_echo(transcript, self.last_tts_text):
            print(f"   ↳ (discarded - echo of TTS)")
            return

        print(f"\n[User]: {transcript}")

        turn_start = time.time()
        self._turn_barged_in = False

        self._set_state(ConversationState.THINKING)
        self.voice_streamer.mute()

        try:
            # Camera frame — physical world context (instant)
            vision_fetch_start = time.time()
            frame = self.framecapture.get_frame()
            vision_fetch_ms = int((time.time() - vision_fetch_start) * 1000)
            print(f"📷 Frame captured in {vision_fetch_ms}ms" if frame else "📷 No frame")

            # Screen context — pre-built by background tracker (instant read)
            workspace_state = self.screen_tracker.get_workspace_state()
            screen_age_s = self.screen_tracker.state_age_s()
            if workspace_state:
                print(f"🖥️  Screen context ({screen_age_s:.0f}s old): {workspace_state[:80]}...")
            else:
                print("🖥️  No screen context yet")

            # Fuse transcript + workspace state into grounded prompt
            prompt = self.fusion.combine(
                transcript=transcript,
                workspace_state=workspace_state or None,
            )

            print("🤔 Thinking (streaming)...")
            self._set_state(ConversationState.SPEAKING)

            llm_start = time.time()
            full_response, first_audio_ms = self._stream_response_to_speech(prompt, frame)
            llm_ms = int((time.time() - llm_start) * 1000)

            print(f"[VISION]: {full_response}")

            self.last_tts_text = full_response
            e2e_ms = int((time.time() - turn_start) * 1000)

            self.eval_logger.log_turn(
                transcript=transcript,
                response=full_response,
                visual_context_preview=workspace_state[:150] if workspace_state else "",
                vision_age_before_s=round(screen_age_s, 1),
                vision_fetch_ms=vision_fetch_ms,
                llm_latency_ms=llm_ms,
                e2e_to_speech_ms=e2e_ms,
                barge_in=self._turn_barged_in,
                time_to_first_audio_ms=first_audio_ms,
                # EOT/ASR timing from the local streamer (empty dict on assemblyai)
                extras=getattr(self.voice_streamer, "last_turn_timing", None),
            )

        finally:
            self._set_state(ConversationState.LISTENING)
            time.sleep(0.3)
            self.voice_streamer.unmute()
            self.audio_frontend.reset()

        print("\n✅ Ready for next input...")

    # ── streaming TTS pipeline ────────────────────────────────────────────────

    def _stream_response_to_speech(self, prompt_text: str, frame) -> tuple[str, int]:
        """
        Stream LLM tokens → buffer into sentences → TTS each sentence in order.

        One sd.RawOutputStream is kept open for the entire turn so there is no
        audio gap between sentences. Audio starts after the first sentence arrives
        (~500ms into LLM generation) rather than after the full response.

        Returns (full_response_text, time_to_first_audio_ms).
        time_to_first_audio_ms is the perceived latency: transcript → first audio byte.
        """
        buffer = ""
        full_response = ""
        stopped = False
        stream_start = time.time()
        first_audio_ms = -1

        self.voice_streamer.start_tts_playback()

        try:
            with sd.RawOutputStream(
                samplerate=self.tts_client.sample_rate,
                channels=1,
                dtype=self.tts_client.audio_format,
            ) as audio_out:

                def speak_chunk(text: str):
                    nonlocal stopped, first_audio_ms
                    if not text.strip() or stopped:
                        return
                    print(f"  🔊 → {text[:70]}{'...' if len(text) > 70 else ''}")
                    for chunk in self.tts_client.stream_sentence(text):
                        if self._check_barge_in():
                            stopped = True
                            return
                        if chunk:
                            if first_audio_ms == -1:
                                first_audio_ms = int((time.time() - stream_start) * 1000)
                                print(f"  ⚡ First audio: {first_audio_ms}ms")
                            self.audio_frontend.feed_tts_reference(chunk)
                            audio_out.write(chunk)

                for token in self.llm_client.generate_streaming(prompt_text, frame):
                    if stopped:
                        break
                    buffer += token
                    full_response += token

                    # Flush any complete sentences from the buffer
                    parts = _SENTENCE_BOUNDARY.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            speak_chunk(sentence)
                            if stopped:
                                break
                        buffer = parts[-1]  # keep partial remainder

                # Speak whatever is left in the buffer
                if buffer.strip() and not stopped:
                    speak_chunk(buffer.strip())

                # Brief silence so PortAudio drains the last real samples before close
                if not stopped:
                    silence = bytes(int(self.tts_client.sample_rate * 0.15) * 2)
                    audio_out.write(silence)

        except Exception as e:
            print(f"❌ Streaming TTS error: {e}")
        finally:
            self.voice_streamer.stop_tts_playback()

        return full_response, first_audio_ms

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self.screen_tracker.start()
        print("\n🚀 VISION is ready!")
        print("Speak into your microphone. Press Ctrl+C to exit.\n")
        self._set_state(ConversationState.LISTENING)
        self.voice_streamer.start_streaming()

    def stop(self):
        print("\nShutting down VISION...")
        self.screen_tracker.stop()
        self.framecapture.release()


def main():
    orchestrator = None
    try:
        orchestrator = Orchestrator()
        orchestrator.start()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if orchestrator:
            orchestrator.stop()


if __name__ == "__main__":
    main()
