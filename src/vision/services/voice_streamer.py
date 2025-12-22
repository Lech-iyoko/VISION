# modules/voice_streamer.py

import os
import logging
from typing import Type
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


class VoiceStreamer:
    def __init__(self, on_final_transcript):
        load_dotenv()
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("ASSEMBLYAI_API_KEY not found in .env file.")

        # Audio settings
        self.sample_rate = 48000
        self.on_final_transcript = on_final_transcript
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
       
        print(f"✅ VoiceStreamer initialized")

    # --- AssemblyAI Event Handlers (v3) ---
   
    def _on_open(self, client: Type[StreamingClient], event: BeginEvent):
        print(f"✅ Connected to AssemblyAI (Session ID: {event.id})")
        print("🎙️ Speak now - listening for audio...")

    def _on_data(self, client: Type[StreamingClient], event: TurnEvent):
        """
        Handles transcript events (Interim and Final).
        """
        text = event.transcript
       
        if not text:
            return

        if event.end_of_turn:
            # Final result
            print(" " * 80, end="\r")
            print(f"📝 FINAL: {text}")
            self.on_final_transcript(text)
        else:
            # Interim result
            print(f"Hearing: {text}...", end="\r")

    def _on_error(self, client: Type[StreamingClient], error: StreamingError):
        print(f"❌ AssemblyAI Error: {error}")

    def _on_close(self, client: Type[StreamingClient], event: TerminationEvent):
        print(f"\n🔌 Connection closed. Audio duration: {event.audio_duration_seconds}s")

    def start_streaming(self):
        """
        Main streaming loop using SDK's MicrophoneStream.
        """
        try:
            print("Setting up AssemblyAI streaming client...")
           
            # 1. Configure the Client
            client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com"
                )
            )

            # 2. Register Callbacks
            client.on(StreamingEvents.Begin, self._on_open)
            client.on(StreamingEvents.Turn, self._on_data)
            client.on(StreamingEvents.Error, self._on_error)
            client.on(StreamingEvents.Termination, self._on_close)

            print("✅ Event handlers registered")
            
            # 3. CRITICAL: Connect to the WebSocket FIRST
            print("🔗 Connecting to AssemblyAI WebSocket...")
            client.connect(
                StreamingParameters(
                    sample_rate=self.sample_rate,
                    encoding="pcm_s16le",
                    format_turns=True,
                    end_of_turn_confidence_threshold=0.6
                )
            )
            
            print("✅ Connected! Starting microphone stream...")
            print(("sample_rate:", self.sample_rate))
            print("🎤 Press Ctrl+C to stop.\n")

            # 4. Stream audio from microphone
            try:
                client.stream(
                    aai.extras.MicrophoneStream(sample_rate=self.sample_rate)
                )
            finally:
                # 5. Always disconnect properly
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


# --- Test Block ---
if __name__ == "__main__":
    def dummy_callback(transcript):
        print(f"\n🎯 TEST CALLBACK RECEIVED: {transcript}")
   
    try:
        streamer = VoiceStreamer(on_final_transcript=dummy_callback)
        streamer.start_streaming()
    except Exception as e:
        print(f"❌ An error occurred: {e}")