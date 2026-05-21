# modules/tts_client.py
import os
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

class ElevenLabsClient:
    def __init__(self, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_turbo_v2"):
        load_dotenv()
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY not found in .env file.")
        
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = 24000
        self.audio_format = "int16"
        print("✅ ElevenLabs Client initialized.")

    def speak_text_stream(self, text_to_speak):
        """
        Generates audio from text and streams it for playback using sounddevice.
        """
        try:
            print(f"🔊 Playing audio: '{text_to_speak}'")
            
            audio_stream = self.client.text_to_speech.stream(
                text=text_to_speak,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=f"pcm_{self.sample_rate}",
            )

            with sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=self.audio_format
            ) as stream:
                for chunk in audio_stream:
                    if chunk:
                        stream.write(chunk)
                        
            print("Finished speaking.")

        except Exception as e:
            print(f"❌ Error playing audio: {e}")

# --- Test Block ---
if __name__ == "__main__":
    try:
        tts_client = ElevenLabsClient()
        test_text = "Hello, this is a test of the ElevenLabs text to speech system."
        tts_client.speak_text_stream(test_text)
    except Exception as e:
        print(f"❌ An error occurred during the test: {e}")