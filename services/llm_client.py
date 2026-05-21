import base64
from typing import Generator
import openai
from io import BytesIO
from PIL import Image
from config import DEFAULT_LLM_MODEL, OLLAMA_BASE_URL, THINKING_ENABLED

_OLLAMA_OPTIONS = {
    "num_ctx": 4096,
    "keep_alive": -1,   # never unload the model between turns
}


class OllamaMultimodalClient:
    def __init__(self, system_prompt: str | None = None):
        self.client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = DEFAULT_LLM_MODEL
        self.system_prompt = system_prompt
        print(f"✅ OllamaMultimodalClient initialized ({self.model})")

    def _build_messages(self, prompt_text: str, image: Image.Image | None = None) -> list:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        content = [{"type": "text", "text": prompt_text}]

        if image is not None:
            buf = BytesIO()
            image.convert("RGB").save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        messages.append({"role": "user", "content": content})
        return messages

    def generate_streaming(
        self, prompt_text: str, image: Image.Image | None = None
    ) -> Generator[str, None, None]:
        """Yield text tokens as they arrive from the model."""
        messages = self._build_messages(prompt_text, image)
        extra = {"options": _OLLAMA_OPTIONS}
        if not THINKING_ENABLED:
            extra["think"] = False
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
            extra_body=extra,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_response(self, prompt_text: str, image: Image.Image | None = None) -> str:
        """Non-streaming fallback — assembles the full response before returning."""
        return "".join(self.generate_streaming(prompt_text, image))


# --- Test Block ---
if __name__ == "__main__":
    client = OllamaMultimodalClient()
    print("Streaming: ", end="", flush=True)
    for token in client.generate_streaming("What is the capital of England? One sentence."):
        print(token, end="", flush=True)
    print()
