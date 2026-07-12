import cv2
import numpy as np
from PIL import Image
import mss


class ScreenCapture:
    """
    Captures the primary display and detects meaningful visual changes
    using pixel-level diff. Acts as the gating layer before any cloud
    VLM call — only changed frames are forwarded for description.
    """

    def __init__(self, diff_threshold: float = 0.02, monitor: int = 1):
        """
        Args:
            diff_threshold: Fraction of pixels that must change to count as
                            a meaningful scene change (default 2%).
            monitor:        mss monitor index — 1 is the primary display.
        """
        self.diff_threshold = diff_threshold
        self.monitor = monitor
        self._prev_gray: np.ndarray | None = None
        self._sct = mss.MSS()
        print(f"✅ ScreenCapture initialized (monitor={monitor}, threshold={diff_threshold:.0%})")

    def capture(self) -> tuple[Image.Image, bool, float]:
        """
        Capture the current screen and compare against the previous frame.

        Returns:
            (frame, changed, diff_score)
            - frame:      PIL Image of the current screen
            - changed:    True if diff_score exceeds the threshold
            - diff_score: Fraction of pixels that changed (0.0–1.0)
        """
        raw = self._sct.grab(self._sct.monitors[self.monitor])
        frame = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame, True, 1.0  # first capture always counts as changed

        diff = cv2.absdiff(gray, self._prev_gray)
        # Threshold: only count pixels that changed by more than 10/255
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        diff_score = float(np.count_nonzero(thresh)) / thresh.size

        changed = diff_score > self.diff_threshold
        if changed:
            self._prev_gray = gray

        return frame, changed, diff_score

    def resize_for_api(self, frame: Image.Image, max_width: int = 1280) -> Image.Image:
        """
        Downscale the frame to reduce token cost before sending to a cloud VLM.
        Maintains aspect ratio.
        """
        if frame.width <= max_width:
            return frame
        ratio = max_width / frame.width
        return frame.resize(
            (max_width, int(frame.height * ratio)),
            Image.LANCZOS,
        )


# --- Test Block ---
if __name__ == "__main__":
    import time

    capture = ScreenCapture(diff_threshold=0.02)

    print("Running 5 captures, 2s apart. Move your mouse or switch windows to trigger changes.")
    for i in range(5):
        frame, changed, score = capture.capture()
        print(f"  [{i+1}] {frame.size} | changed={changed} | diff={score:.3%}")
        time.sleep(2)
