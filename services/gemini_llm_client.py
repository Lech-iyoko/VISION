from typing import Generator
from PIL import Image
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_LLM_MODEL, THINKING_ENABLED


class GeminiLLMClient:
    """
    Drop-in replacement for OllamaMultimodalClient using Gemini 2.5 Flash.
    Accepts text prompts and optional PIL Image camera frames natively.
    Same interface: generate_streaming(prompt_text, image) → token generator.
    """

    def __init__(self, system_prompt: str | None = None):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = GEMINI_LLM_MODEL
        self._system_prompt = system_prompt
        print(f"✅ GeminiLLMClient initialized ({self._model})")

    def _config(self) -> types.GenerateContentConfig:
        thinking_cfg = (
            None if THINKING_ENABLED
            else types.ThinkingConfig(thinking_budget=0)
        )
        return types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=0.7,
            max_output_tokens=1024,
            thinking_config=thinking_cfg,
        )

    def generate_streaming(
        self, prompt_text: str, image: Image.Image | None = None
    ) -> Generator[str, None, None]:
        contents = []
        if image is not None:
            contents.append(image)
        contents.append(prompt_text)

        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=self._config(),
        ):
            if chunk.text:
                yield chunk.text

    def generate_response(self, prompt_text: str, image: Image.Image | None = None) -> str:
        return "".join(self.generate_streaming(prompt_text, image))


if __name__ == "__main__":
    client = GeminiLLMClient()
    print("Streaming: ", end="", flush=True)
    for token in client.generate_streaming("What is the capital of England? One sentence."):
        print(token, end="", flush=True)
    print()
