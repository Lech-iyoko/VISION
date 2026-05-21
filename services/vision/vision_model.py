import base64
from io import BytesIO
import requests
from config import DEFAULT_VLM_MODEL

# Ollama native endpoint — the OpenAI-compatible endpoint does not support
# moondream's image format and causes a GGML_ASSERT crash.
_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


class VisionModel:
    """
    Wrapper for moondream served via Ollama.
    Runs on CPU (num_gpu=0) so qwen3:8b can stay resident in VRAM.
    Takes PIL images and returns text descriptions.
    """

    def __init__(self, model_name: str = DEFAULT_VLM_MODEL):
        self.model_name = model_name
        print(f"✅ VisionModel initialized ({model_name}, CPU inference)")

    def describe_image(self, image, prompt: str = None) -> str:
        """
        Send a PIL image to moondream via Ollama and return a description.

        Args:
            image: PIL Image object
            prompt: Custom prompt (optional)

        Returns:
            String description of the image
        """
        if prompt is None:
            prompt = "Describe what you see in this image concisely. Focus on people, objects, and actions."

        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            response = requests.post(
                _OLLAMA_GENERATE_URL,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"num_gpu": 0},  # Keep qwen3 in VRAM
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            print(f"❌ VisionModel error: {e}")
            return ""


# --- Test Block ---
if __name__ == "__main__":
    from PIL import Image

    test_image = Image.new('RGB', (100, 100), color='red')
    model = VisionModel()
    description = model.describe_image(test_image)
    print(f"Description: {description}")
