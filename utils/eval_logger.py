import json
from datetime import datetime
from pathlib import Path


class EvaluationLogger:
    """
    Writes one JSONL record per conversation turn to eval_logs/.
    Used to build a baseline for comparing cloud vs local model implementations.

    Each record captures:
      - Timing: vision fetch, LLM inference, end-to-end to first speech
      - Quality signals: transcript, response, visual context snapshot
      - Context: model names, session ID, whether the turn was barged into
    """

    def __init__(self, llm_model: str, vlm_model: str, log_dir: str = "evals"):
        self.llm_model = llm_model
        self.vlm_model = vlm_model

        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_path / f"session_{self.session_id}.jsonl"
        self._turn = 0

        print(f"📊 EvalLogger → {self.log_file}")

    def log_turn(
        self,
        transcript: str,
        response: str,
        visual_context_preview: str,
        vision_age_before_s: float,
        vision_fetch_ms: int,
        llm_latency_ms: int,
        e2e_to_speech_ms: int,
        barge_in: bool,
        time_to_first_audio_ms: int = -1,
    ) -> None:
        self._turn += 1
        record = {
            "session_id": self.session_id,
            "turn": self._turn,
            "ts": datetime.now().isoformat(),
            "llm_model": self.llm_model,
            "vlm_model": self.vlm_model,
            "transcript": transcript,
            "response": response,
            "visual_context_preview": visual_context_preview[:150] if visual_context_preview else "",
            "vision_age_before_s": vision_age_before_s,
            "vision_fetch_ms": vision_fetch_ms,
            "llm_latency_ms": llm_latency_ms,
            "time_to_first_audio_ms": time_to_first_audio_ms,
            "e2e_to_speech_ms": e2e_to_speech_ms,
            "barge_in": barge_in,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
