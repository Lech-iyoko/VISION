"""
Semantic end-of-turn detection — pipecat smart-turn v3.

An 8M-parameter classifier (Whisper-Tiny encoder + linear head) that predicts
whether the speaker has finished their turn from the raw waveform — prosody
and intonation, not the transcript. Falling pitch on a complete phrase scores
high; a mid-thought pause or rising pitch scores low.

Runs on CPU in ~10-30ms, so it can be called during live silence without
touching the GPU that Whisper occupies.

Model: https://huggingface.co/pipecat-ai/smart-turn-v3 (BSD-2, open weights)
Inference mirrors the reference implementation in
https://github.com/pipecat-ai/smart-turn (inference.py).
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

_REPO_ID = "pipecat-ai/smart-turn-v3"
_MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
_SAMPLE_RATE = 16000
_MAX_SECONDS = 8  # model input window; longer audio is truncated to the last 8s


class TurnDetector:
    """
    predict(audio) -> P(turn complete) for 16kHz mono float32 audio.

    The model was trained on 8-second windows with the speech at the end,
    so we keep the most recent 8 seconds and let the Whisper feature
    extractor handle padding for shorter segments.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

        print(f"Loading smart-turn v3 ({_MODEL_FILE})...")
        model_path = hf_hub_download(repo_id=_REPO_ID, filename=_MODEL_FILE)

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        self._extractor = WhisperFeatureExtractor(chunk_length=_MAX_SECONDS)

        # Warm-up: first call pays one-time allocation costs
        self.predict(np.zeros(_SAMPLE_RATE, dtype=np.float32))
        print(f"✅ TurnDetector ready (threshold={threshold})")

    def predict(self, audio: np.ndarray) -> float:
        """Returns P(speaker has finished their turn), 0.0-1.0."""
        audio = audio[-_SAMPLE_RATE * _MAX_SECONDS:].astype(np.float32)

        features = self._extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            padding="max_length",
            max_length=_SAMPLE_RATE * _MAX_SECONDS,
            truncation=True,
            do_normalize=True,
            return_tensors="np",
        ).input_features.astype(np.float32)

        outputs = self._session.run(None, {self._input_name: features})
        prob = float(np.asarray(outputs[0]).squeeze())

        # The graph ends in a sigmoid; guard in case a model revision exports logits
        if prob < 0.0 or prob > 1.0:
            prob = 1.0 / (1.0 + np.exp(-prob))
        return prob

    def is_complete(self, audio: np.ndarray) -> bool:
        return self.predict(audio) >= self.threshold


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    detector = TurnDetector()

    print("\nTiming 5 predictions on 3s of noise...")
    audio = (np.random.randn(3 * _SAMPLE_RATE) * 0.05).astype(np.float32)
    for i in range(5):
        t0 = time.time()
        prob = detector.predict(audio)
        print(f"  [{i+1}] P(complete)={prob:.3f}  ({int((time.time()-t0)*1000)}ms)")
