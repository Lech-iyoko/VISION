import numpy as np
from typing import Iterator
from kokoro import KPipeline
from config import KOKORO_VOICE


class KokoroTTSClient:
    """
    Local neural TTS using Kokoro-82M.
    Matches ElevenLabsClient interface: stream_sentence(text) -> Iterator[bytes].
    Output: 24kHz mono int16 PCM — same sample rate as ElevenLabs.
    """

    sample_rate = 24000
    audio_format = "int16"

    def __init__(self):
        self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
        self._voice = KOKORO_VOICE
        print(f"✅ KokoroTTSClient initialized (voice={self._voice})")

    def stream_sentence(self, text: str) -> Iterator[bytes]:
        """Yields raw PCM int16 chunks — same interface as ElevenLabsClient."""
        for _, _, audio in self._pipeline(text, voice=self._voice, speed=1.0):
            if audio is not None and len(audio) > 0:
                yield (audio * 32767).cpu().numpy().astype(np.int16).tobytes()


if __name__ == "__main__":
    import sounddevice as sd

    client = KokoroTTSClient()
    text = "Good morning, sir. VISION is online and ready."
    print(f"Speaking: {text!r}")

    with sd.RawOutputStream(
        samplerate=client.sample_rate,
        channels=1,
        dtype=client.audio_format,
    ) as stream:
        for chunk in client.stream_sentence(text):
            stream.write(chunk)
    print("Done.")
