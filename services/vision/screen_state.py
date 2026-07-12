import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()

from config import (
    GEMINI_API_KEY,
    SCREEN_CAPTURE_INTERVAL_S,
    SCREEN_DIFF_THRESHOLD,
    SCREEN_GEMINI_MODEL,
    SCREEN_STATE_BUFFER_SIZE,
)
from services.vision.screen_capture import ScreenCapture

# Keywords that elevate a description to a proactive trigger.
_SIGNAL_KEYWORDS = (
    "error", "exception", "traceback", "failed", "failure",
    "warning", "undefined", "null", "segfault", "crash",
    "permission denied", "not found", "syntax",
)

_DESCRIBE_PROMPT = (
    "You are the visual sensor for an AI assistant. "
    "Describe what is on this screen in 2–3 sentences. "
    "Focus on: what application is open, any visible code or terminal output, "
    "any error messages or warnings, and what the user appears to be doing. "
    "Do not describe UI chrome or window decorations. Be specific and concise."
)


@dataclass
class ScreenEvent:
    timestamp: float
    description: str
    diff_score: float
    gemini_latency_ms: int


class ScreenStateTracker:
    """
    Background thread that:
      1. Captures the screen every SCREEN_CAPTURE_INTERVAL_S seconds.
      2. Uses OpenCV diff to skip unchanged frames (no Gemini call).
      3. On meaningful change: calls Gemini 2.5 Flash, stores the description.
      4. Maintains a rolling buffer of recent screen events.
      5. Fires an optional callback when a high-signal event is detected
         (errors, exceptions, crashes) — this is the proactive trigger.
    """

    def __init__(
        self,
        on_proactive_trigger: Optional[Callable[[str], None]] = None,
    ):
        self._capture = ScreenCapture(diff_threshold=SCREEN_DIFF_THRESHOLD)
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._buffer: deque[ScreenEvent] = deque(maxlen=SCREEN_STATE_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_proactive = on_proactive_trigger
        print(f"✅ ScreenStateTracker initialized ({SCREEN_GEMINI_MODEL})")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="screen-state"
        )
        self._thread.start()
        print("🖥️  Screen state loop started")

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        print("🖥️  Screen state loop stopped")

    # ── background loop ───────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(SCREEN_CAPTURE_INTERVAL_S)

    def _tick(self):
        try:
            frame, changed, diff_score = self._capture.capture()

            if not changed:
                return

            resized = self._capture.resize_for_api(frame)

            t0 = time.time()
            response = self._client.models.generate_content(
                model=SCREEN_GEMINI_MODEL,
                contents=[_DESCRIBE_PROMPT, resized],
            )
            latency_ms = int((time.time() - t0) * 1000)

            description = response.text.strip()
            event = ScreenEvent(
                timestamp=time.time(),
                description=description,
                diff_score=diff_score,
                gemini_latency_ms=latency_ms,
            )

            with self._lock:
                self._buffer.append(event)

            print(f"🖥️  [{latency_ms}ms | diff={diff_score:.1%}] {description[:120]}")

            if self._is_high_signal(description) and self._on_proactive:
                self._on_proactive(description)

        except Exception as e:
            print(f"❌ ScreenStateTracker error: {e}")

    def _is_high_signal(self, description: str) -> bool:
        lower = description.lower()
        return any(kw in lower for kw in _SIGNAL_KEYWORDS)

    # ── public API ────────────────────────────────────────────────────────────

    def get_workspace_state(self) -> str:
        """
        Returns a temporal summary of recent screen activity for the LLM.
        Format: chronological list of what was seen, oldest to newest.
        Returns empty string if no events recorded yet.
        """
        with self._lock:
            events = list(self._buffer)

        if not events:
            return ""

        now = time.time()
        lines = []
        for e in events:
            age_s = int(now - e.timestamp)
            lines.append(f"{age_s}s ago: {e.description}")

        return "\n".join(lines)

    def latest_description(self) -> str:
        with self._lock:
            return self._buffer[-1].description if self._buffer else ""

    def state_age_s(self) -> float:
        with self._lock:
            if not self._buffer:
                return float("inf")
            return time.time() - self._buffer[-1].timestamp


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("SCREEN PIPELINE TEST")
    print("=" * 60)
    print()

    # Step 1: verify Gemini connection with a text-only call
    print("Step 1 — Gemini API connection check...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    t0 = time.time()
    r = client.models.generate_content(
        model=SCREEN_GEMINI_MODEL,
        contents="Reply with exactly: connection ok",
    )
    latency = int((time.time() - t0) * 1000)
    print(f"  Response: '{r.text.strip()}' ({latency}ms)")
    print()

    # Step 2: single screen capture + Gemini description
    print("Step 2 — Screen capture + Gemini description...")
    capture = ScreenCapture(diff_threshold=SCREEN_DIFF_THRESHOLD)
    frame, _, diff_score = capture.capture()
    resized = capture.resize_for_api(frame)
    print(f"  Captured: {frame.size} → resized to {resized.size}")

    t0 = time.time()
    response = client.models.generate_content(
        model=SCREEN_GEMINI_MODEL,
        contents=[_DESCRIBE_PROMPT, resized],
    )
    latency = int((time.time() - t0) * 1000)
    description = response.text.strip()

    print(f"  Gemini latency: {latency}ms")
    print(f"  Description:")
    print(f"    {description}")
    print()

    # Step 3: diff gating — capture again and check if Gemini is skipped
    print("Step 3 — Diff gating (no screen change expected)...")
    frame2, changed, diff_score2 = capture.capture()
    print(f"  diff_score={diff_score2:.3%} | changed={changed}")
    if not changed:
        print("  ✅ Gemini call correctly skipped — no meaningful change")
    else:
        print("  ℹ️  Screen changed — Gemini would fire")
    print()

    # Step 4: background loop for 20 seconds
    print("Step 4 — Background loop (20s). Change your screen to trigger Gemini calls.")
    print("         (switch windows, open terminal, type something)")
    print()

    def on_trigger(desc: str):
        print(f"  🚨 PROACTIVE TRIGGER: {desc[:100]}")

    tracker = ScreenStateTracker(on_proactive_trigger=on_trigger)
    tracker.start()
    time.sleep(20)
    tracker.stop()

    print()
    print("Workspace state summary:")
    print(tracker.get_workspace_state() or "  (no events captured)")
