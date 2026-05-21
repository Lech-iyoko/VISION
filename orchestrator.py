"""
Orchestrator - The central coordinator for VISION.

Audio pipeline:
  VoiceStreamer (ASR) → AudioFrontend (AEC + VAD) → OllamaMultimodalClient (LLM+vision) → ElevenLabsClient (TTS)

Vision:
  FrameCapture — captures one fresh frame at turn time

Conversation state machine:
  IDLE → LISTENING → THINKING → SPEAKING → IDLE

Latency strategy: stream LLM tokens, buffer into sentences, TTS each sentence while
the next one is still being generated. Audio begins after the first sentence (~500ms
of LLM output) rather than after the full response.
"""

import re
import time
import threading
import sounddevice as sd
from services.voice.voice_streamer import VoiceStreamer
from services.voice.audio_frontend import AudioFrontend, AudioConfig
from services.llm_client import OllamaMultimodalClient
from services.voice.tts_client import ElevenLabsClient
from services.voice.echo_guard import is_echo
from services.vision.frame_capture import FrameCapture

from dotenv import load_dotenv
from config import DEFAULT_VOICE_ID, SYSTEM_PROMPT, DEFAULT_LLM_MODEL
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
        self.llm_client = OllamaMultimodalClient(system_prompt=SYSTEM_PROMPT)
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

        # === Vision — single frame at turn time ===
        self.framecapture = FrameCapture(camera_index=0)

        # Evaluation logger
        self.eval_logger = EvaluationLogger(
            llm_model=DEFAULT_LLM_MODEL,
            vlm_model="none",
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
            # Capture one fresh frame at the moment of the turn
            vision_fetch_start = time.time()
            frame = self.framecapture.get_frame()
            vision_fetch_ms = int((time.time() - vision_fetch_start) * 1000)
            print(f"📷 Frame captured in {vision_fetch_ms}ms" if frame else "📷 No frame")

            print("🤔 Thinking (streaming)...")
            self._set_state(ConversationState.SPEAKING)

            llm_start = time.time()
            full_response, first_audio_ms = self._stream_response_to_speech(transcript, frame)
            llm_ms = int((time.time() - llm_start) * 1000)

            print(f"[VISION]: {full_response}")

            self.last_tts_text = full_response
            e2e_ms = int((time.time() - turn_start) * 1000)

            self.eval_logger.log_turn(
                transcript=transcript,
                response=full_response,
                visual_context_preview="[frame attached]" if frame else "",
                vision_age_before_s=0,
                vision_fetch_ms=vision_fetch_ms,
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
                    audio_gen = self.tts_client.client.text_to_speech.stream(
                        text=text,
                        voice_id=self.tts_client.voice_id,
                        model_id=self.tts_client.model_id,
                        output_format=f"pcm_{self.tts_client.sample_rate}",
                    )
                    for chunk in audio_gen:
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
        print("\n🚀 VISION is ready!")
        print("Speak into your microphone. Press Ctrl+C to exit.\n")
        self._set_state(ConversationState.LISTENING)
        self.voice_streamer.start_streaming()

    def stop(self):
        print("\nShutting down VISION...")
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
