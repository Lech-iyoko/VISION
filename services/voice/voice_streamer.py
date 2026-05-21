# modules/voice_streamer.py

import os
import logging
import time
import threading
import numpy as np
from typing import Type, Callable
from dotenv import load_dotenv

# AssemblyAI v3 Imports
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingEvents,
    StreamingError
)

# Silero VAD
import torch
torch.set_num_threads(1)  # Optimize for real-time


class VoiceStreamer:
    def __init__(self, on_final_transcript: Callable[[str], None]):
        load_dotenv()
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("ASSEMBLYAI_API_KEY not found in .env file.")

        # Audio settings
        self.sample_rate = 16000
        self.on_final_transcript = on_final_transcript

        # === Echo Prevention State ===
        self.is_muted = False
        self.unmute_time = 0
        self.cooldown_seconds = 1.5  # Ignore transcripts for this long after unmute

        # === Silero VAD Setup ===
        self._setup_vad()
        
        # VAD state
        self.vad_lock = threading.Lock()
        self.user_is_speaking = False
        self.speech_start_time = 0
        self.min_speech_duration = 0.5  # INCREASED: 500ms of sustained speech for barge-in
        
        # VAD audio buffer (Silero needs exactly 512 samples at 16kHz)
        self.vad_chunk_size = 512
        self.vad_buffer = np.array([], dtype=np.float32)
        
        # VAD sensitivity settings
        self.vad_threshold = 0.7  # INCREASED: Higher threshold = less sensitive (was 0.5)
        self.consecutive_speech_frames = 0
        self.required_speech_frames = 5  # Need 5 consecutive frames (~160ms) before considering it speech
        
        # === TTS playback state (only detect barge-in during actual playback) ===
        self.tts_is_playing = False
        self.barge_in_triggered = False  # Prevent multiple triggers
        
        # Barge-in callback (set by orchestrator)
        self.on_barge_in: Callable[[], None] | None = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
       
        print("✅ VoiceStreamer initialized with Silero VAD")
        print(f"   VAD threshold: {self.vad_threshold}")
        print(f"   Min speech duration: {self.min_speech_duration}s")

    def _setup_vad(self):
        """Initialize Silero VAD model using silero-vad package."""
        print("Loading Silero VAD model...")
        try:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad(onnx=True)
            print("✅ Silero VAD loaded (ONNX mode)")
        except ImportError:
            print("silero-vad package not found, using torch.hub...")
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            print("✅ Silero VAD loaded via torch.hub")

    def process_audio_for_vad(self, audio_bytes: bytes) -> list[bool]:
        """
        Process audio chunk through VAD.
        Returns list of speech detection results (one per 512-sample chunk).
        """
        results = []
        
        try:
            # Convert bytes to float32
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Add to buffer
            self.vad_buffer = np.concatenate([self.vad_buffer, audio_float32])
            
            # Process complete 512-sample chunks
            while len(self.vad_buffer) >= self.vad_chunk_size:
                chunk = self.vad_buffer[:self.vad_chunk_size]
                self.vad_buffer = self.vad_buffer[self.vad_chunk_size:]
                
                # Convert to tensor and get speech probability
                audio_tensor = torch.from_numpy(chunk)
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                
                # Use higher threshold
                is_speech = speech_prob > self.vad_threshold
                results.append(is_speech)
                
        except Exception as e:
            self.logger.error(f"VAD error: {e}")
            
        return results

    def _handle_vad_speech_detected(self):
        """Called when VAD detects speech during TTS playback."""
        with self.vad_lock:
            # Only process if TTS is actually playing and we haven't already triggered
            if not self.tts_is_playing or self.barge_in_triggered:
                return
                
            current_time = time.time()
            
            # Increment consecutive speech frame counter
            self.consecutive_speech_frames += 1
            
            # Need sustained speech before we consider it real
            if self.consecutive_speech_frames < self.required_speech_frames:
                return  # Not enough consecutive frames yet
            
            if not self.user_is_speaking:
                # Speech just started (after meeting frame threshold)
                self.user_is_speaking = True
                self.speech_start_time = current_time
                print("🎤 VAD: Sustained speech detected...")
            else:
                # Speech continuing - check duration for barge-in
                speech_duration = current_time - self.speech_start_time
                
                if speech_duration >= self.min_speech_duration:
                    print(f"🛑 BARGE-IN triggered! (speech: {speech_duration:.2f}s)")
                    self.barge_in_triggered = True  # Prevent multiple triggers
                    if self.on_barge_in:
                        self.on_barge_in()

    def _handle_vad_silence_detected(self):
        """Called when VAD detects silence."""
        with self.vad_lock:
            # Reset consecutive speech counter
            self.consecutive_speech_frames = 0
            
            if self.user_is_speaking:
                self.user_is_speaking = False

    def start_tts_playback(self):
        """Call this when TTS starts playing."""
        with self.vad_lock:
            self.tts_is_playing = True
            self.barge_in_triggered = False
            self.user_is_speaking = False
            self.consecutive_speech_frames = 0
            self.vad_buffer = np.array([], dtype=np.float32)
            if hasattr(self.vad_model, 'reset_states'):
                self.vad_model.reset_states()
        print("🔊 TTS playback started - barge-in detection active")

    def stop_tts_playback(self):
        """Call this when TTS stops playing."""
        with self.vad_lock:
            self.tts_is_playing = False
            self.barge_in_triggered = False
            self.user_is_speaking = False
            self.consecutive_speech_frames = 0
        print("🔇 TTS playback stopped - barge-in detection paused")

    def mute(self):
        """Mute transcript processing (call before TTS plays)."""
        self.is_muted = True
        self.vad_buffer = np.array([], dtype=np.float32)
        if hasattr(self.vad_model, 'reset_states'):
            self.vad_model.reset_states()
        print("🔇 Transcript processing MUTED")

    def unmute(self):
        """Unmute transcript processing (call after TTS finishes)."""
        self.is_muted = False
        self.unmute_time = time.time()
        self.user_is_speaking = False
        self.consecutive_speech_frames = 0
        self.vad_buffer = np.array([], dtype=np.float32)
        if hasattr(self.vad_model, 'reset_states'):
            self.vad_model.reset_states()
        print(f"🎤 Transcript processing UNMUTED (cooldown: {self.cooldown_seconds}s)")

    # --- AssemblyAI Event Handlers (v3) ---

    def _on_open(self, client: Type[StreamingClient], event: BeginEvent):
        print(f"✅ Connected to AssemblyAI (Session ID: {event.id})")
        print("🎙️ Speak now - listening for audio...")

    def _on_data(self, client: Type[StreamingClient], event: TurnEvent):
        """Handles transcript events (Interim and Final)."""
        # Raw debug — shows every event regardless of content
        print(f"[ASR] event received | text='{event.transcript}' | end_of_turn={event.end_of_turn}")

        text = event.transcript

        if not text:
            return

        if event.end_of_turn:
            print(" " * 80, end="\r")
            print(f"📝 FINAL: {text}")
            
            if self.is_muted:
                print("   ↳ (ignored - muted)")
                return
            
            time_since_unmute = time.time() - self.unmute_time
            if time_since_unmute < self.cooldown_seconds:
                print(f"   ↳ (ignored - cooldown: {time_since_unmute:.1f}s < {self.cooldown_seconds}s)")
                return
            
            self.on_final_transcript(text)
        else:
            if not self.is_muted:
                time_since_unmute = time.time() - self.unmute_time
                if time_since_unmute >= self.cooldown_seconds:
                    print(f"Hearing: {text}...", end="\r")

    def _on_error(self, client: Type[StreamingClient], error: StreamingError):
        print(f"❌ AssemblyAI Error: {error}")

    def _on_close(self, client: Type[StreamingClient], event: TerminationEvent):
        print(f"\n🔌 Connection closed. Audio duration: {event.audio_duration_seconds}s")

    def start_streaming(self):
        """Main streaming loop with VAD integration."""
        try:
            print("Setting up AssemblyAI streaming client...")
           
            client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com"
                )
            )

            client.on(StreamingEvents.Begin, self._on_open)
            client.on(StreamingEvents.Turn, self._on_data)
            client.on(StreamingEvents.Error, self._on_error)
            client.on(StreamingEvents.Termination, self._on_close)

            print("✅ Event handlers registered")
            print("🔗 Connecting to AssemblyAI WebSocket...")
            
            client.connect(
                StreamingParameters(
                    sample_rate=self.sample_rate,
                    encoding="pcm_s16le",
                    format_turns=True,
                    end_of_turn_confidence_threshold=0.6,
                    speech_model="universal-streaming-english",
                )
            )
            
            print("✅ Connected! Starting microphone stream...")
            print(f"📊 Sample rate: {self.sample_rate} Hz")
            print("🎤 Press Ctrl+C to stop.\n")

            try:
                mic_stream = VADMicrophoneStream(
                    sample_rate=self.sample_rate,
                    vad_callback=self._process_vad_chunk
                )
                client.stream(mic_stream)
            finally:
                print("\n🛑 Disconnecting...")
                client.disconnect(terminate=True)

        except KeyboardInterrupt:
            print("\n⏸️ Stream interrupted by user")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("✅ Stream finished.")

    def _process_vad_chunk(self, audio_bytes: bytes):
        """Process audio chunk for VAD (called by custom mic stream)."""
        # Audio level monitor — remove once mic is confirmed working
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        if rms > 200:  # Only print when there's actual signal
            print(f"[MIC] RMS={rms:.0f}", end="\r")

        # Only run VAD when TTS is playing (to detect barge-in)
        if self.tts_is_playing and not self.barge_in_triggered:
            speech_results = self.process_audio_for_vad(audio_bytes)
            
            for is_speech in speech_results:
                if is_speech:
                    self._handle_vad_speech_detected()
                else:
                    self._handle_vad_silence_detected()


class VADMicrophoneStream:
    """Custom microphone stream that also processes audio through VAD."""
    
    def __init__(self, sample_rate: int, vad_callback: Callable[[bytes], None]):
        self.sample_rate = sample_rate
        self.vad_callback = vad_callback
        self._mic_stream = aai.extras.MicrophoneStream(sample_rate=sample_rate)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> bytes:
        chunk = next(self._mic_stream)
        self.vad_callback(chunk)
        return chunk
    
    def close(self):
        if hasattr(self._mic_stream, 'close'):
            self._mic_stream.close()


# --- Test Block ---
if __name__ == "__main__":
    def dummy_callback(transcript):
        print(f"\n🎯 TEST CALLBACK RECEIVED: {transcript}")
   
    try:
        streamer = VoiceStreamer(on_final_transcript=dummy_callback)
        streamer.start_streaming()
    except Exception as e:
        print(f"❌ An error occurred: {e}")