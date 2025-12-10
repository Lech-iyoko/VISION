from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1SocketClientResponse, ListenV1MediaMessage
from dotenv import load_dotenv
import os
import sounddevice as sd
import numpy as np
import threading
import time

load_dotenv()

print("Testing Deepgram connection with microphone...")

client = DeepgramClient(
    api_key=os.getenv("DEEPGRAM_API_KEY"),
)

print("Client created. Attempting to connect...")

# Microphone settings
SAMPLE_RATE = 16000  # Deepgram recommends 16kHz
DEVICE_INDEX = None  # None = default mic

try:
    with client.listen.v1.connect(
        model="nova-2",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        punctuate=True,
        interim_results=True,
    ) as connection:
        
        def on_message(message: ListenV1SocketClientResponse) -> None:
            msg_type = getattr(message, "type", "Unknown")
            
            if msg_type == "Results":
                # Extract transcript
                try:
                    transcript = message.channel.alternatives[0].transcript
                    if transcript.strip():
                        is_final = message.is_final
                        prefix = "FINAL" if is_final else "interim"
                        print(f"📝 [{prefix}]: {transcript}")
                except:
                    pass
            else:
                print(f"✅ Received {msg_type} event")

        connection.on(EventType.OPEN, lambda _: print("✅ Connection opened"))
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.CLOSE, lambda _: print("❌ Connection closed"))
        connection.on(EventType.ERROR, lambda error: print(f"❌ Error: {error}"))

        # Audio callback to send mic data
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            # Convert to bytes and send to Deepgram
            audio_bytes = indata.flatten().tobytes()
            connection.send_media(ListenV1MediaMessage(audio_bytes))

        # Start listening
        print("Starting to listen...")
        connection.start_listening()
        
        # Start microphone streaming
        print(f"🎤 Starting microphone (speak now for 10 seconds)...")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            callback=audio_callback,
            device=DEVICE_INDEX
        ):
            time.sleep(10)  # Record for 10 seconds
        
        print("Test complete!")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
