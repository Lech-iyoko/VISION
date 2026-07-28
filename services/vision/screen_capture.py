import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ScreenCaptureError(Exception):
    """Raised when the underlying screenshot tool fails or times out."""


class ScreenCapture:
    """
    Captures the primary display and detects meaningful visual changes
    using pixel-level diff. Acts as the gating layer before any cloud
    VLM call — only changed frames are forwarded for description.

    Acquisition uses `cosmic-screenshot`, COSMIC's portal-based screenshot
    CLI — not `mss`. mss reads the X11 root window, which native Wayland
    (COSMIC's compositor, cosmic-comp) does not populate: it silently
    returns a static black frame instead of raising, which made the
    pixel-diff gate below "correctly" conclude nothing ever changes.
    cosmic-screenshot goes through the real Wayland screenshot portal.

    This hard-requires COSMIC/cosmic-screenshot rather than falling back
    to mss on failure — a silent fallback here would just reintroduce the
    same black-frame bug under a different name.
    """

    _CAPTURE_TIMEOUT_S = 3.0  # generous vs. the ~70ms measured cost

    def __init__(self, diff_threshold: float = 0.02, monitor: int = 1):
        """
        Args:
            diff_threshold: Fraction of pixels that must change to count as
                            a meaningful scene change (default 2%).
            monitor:        unused (single-monitor capture only for now);
                            kept for interface compatibility.
        """
        self.diff_threshold = diff_threshold
        self.monitor = monitor
        self._prev_gray: np.ndarray | None = None

        # cosmic-screenshot names its own output file — it doesn't take an
        # exact output path — so freshness comes from owning this directory
        # exclusively, not from the filename. Cleared now so a file left
        # behind by a crashed prior run can never be mistaken for a fresh
        # capture on the first call.
        self._capture_dir = Path(tempfile.gettempdir()) / "vision_screencap"
        if self._capture_dir.exists():
            shutil.rmtree(self._capture_dir)
        self._capture_dir.mkdir(parents=True)

        print(f"✅ ScreenCapture initialized (backend=cosmic-screenshot, threshold={diff_threshold:.0%})")

    def _grab(self) -> Image.Image:
        """Capture the screen via cosmic-screenshot and load the result.

        The capture directory is ours exclusively: empty before this call,
        exactly one file after. That's what guarantees the file we read is
        this call's fresh capture and not a leftover from a previous one —
        not the filename (which cosmic-screenshot chooses, not us).
        """
        try:
            subprocess.run(
                [
                    "cosmic-screenshot",
                    "--interactive=false",
                    "--modal=false",
                    "--notify=false",
                    f"--save-dir={self._capture_dir}",
                ],
                timeout=self._CAPTURE_TIMEOUT_S,
                check=True,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
            # OSError also catches FileNotFoundError (binary missing/not on PATH)
            raise ScreenCaptureError(f"cosmic-screenshot failed: {e}") from e

        produced = list(self._capture_dir.glob("*.png"))
        try:
            if len(produced) != 1:
                raise ScreenCaptureError(
                    f"expected exactly one screenshot, found {len(produced)}"
                )
            image = Image.open(produced[0]).convert("RGB")
            image.load()  # force the read before the file is deleted below
            return image
        finally:
            for f in produced:
                f.unlink(missing_ok=True)

    def capture(self) -> tuple[Image.Image, bool, float]:
        """
        Capture the current screen and compare against the previous frame.

        Returns:
            (frame, changed, diff_score)
            - frame:      PIL Image of the current screen
            - changed:    True if diff_score exceeds the threshold
            - diff_score: Fraction of pixels that changed (0.0–1.0)
        """
        frame = self._grab()

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
