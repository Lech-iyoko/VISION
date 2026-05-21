import time
import threading
from typing import Optional


class VisionManager:
    """
    Runs a background loop that continuously captures frames and updates
    the visual context cache. get_visual_context() is always instant —
    no blocking API calls on the conversation path.
    """

    def __init__(self, frame_capture, vision_model, interval: float = 1.5):
        """
        Args:
            frame_capture: FrameCapture instance
            vision_model:  VisionModel instance
            interval:      Seconds between captures (default 4s)
        """
        self.frame_capture = frame_capture
        self.vision_model = vision_model
        self.interval = interval

        self._latest_context: str = ""
        self._latest_timestamp: float = 0.0
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._stale_threshold: float = 2.0
        print(f"✅ VisionManager initialized (interval: {interval}s)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background vision loop."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vision-loop"
        )
        self._thread.start()
        print("👁️  Vision loop started")

    def stop(self):
        """Stop the background vision loop and wait for it to exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6.0)
        print("👁️  Vision loop stopped")

    def release(self):
        """Stop the loop and release camera resources."""
        self.stop()
        if hasattr(self.frame_capture, "release"):
            self.frame_capture.release()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self):
        """Runs in a daemon thread: capture → describe → cache → sleep."""
        while not self._stop_event.is_set():
            self._update()
            # Wait for the next interval, but wake immediately if stop() is called.
            self._stop_event.wait(self.interval)

    def _update(self):
        """Single capture-and-describe cycle."""
        try:
            frame = self.frame_capture.get_frame()
            if frame is None:
                return

            description = self.vision_model.describe_image(frame)
            if description:
                with self._lock:
                    self._latest_context = description
                    self._latest_timestamp = time.time()
                print(f"👁️  [Vision] {description[:100]}...")

        except Exception as e:
            print(f"❌ Vision loop error: {e}")

    # ------------------------------------------------------------------
    # Public API (non-blocking reads)
    # ------------------------------------------------------------------

    def get_visual_context(self, ensure_fresh: bool = False) -> str:
        """
        Return the latest visual context.

        If ensure_fresh=True and the cache is older than _stale_threshold,
        blocks for a synchronous capture before returning.
        """
        if ensure_fresh and self.context_age() > self._stale_threshold:
            self._update()
        with self._lock:
            return self._latest_context

    def context_age(self) -> float:
        """Seconds since the last successful vision update."""
        with self._lock:
            if self._latest_timestamp == 0.0:
                return float("inf")
            return time.time() - self._latest_timestamp

    def force_update(self):
        """Trigger an immediate capture outside of the normal interval."""
        self._update()


# --- Test Block ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from frame_capture import FrameCapture
    from vision_model import VisionModel

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    capture = FrameCapture()
    model = VisionModel(api_key=api_key)
    manager = VisionManager(frame_capture=capture, vision_model=model, interval=3.0)

    manager.start()
    print("Waiting for first capture...")
    time.sleep(5)
    print(f"Context: {manager.get_visual_context()}")
    print(f"Age: {manager.context_age():.1f}s")
    manager.release()
