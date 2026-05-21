import cv2
from PIL import Image


class FrameCapture:
    """
    Captures frames from a webcam and returns them as PIL Images.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal buffer depth
        
        if not self.cap.isOpened():
            print(f"⚠️ Warning: Could not open camera {camera_index}")
        else:
            print(f"✅ FrameCapture initialized (camera {camera_index}, {width}x{height})")

    def get_frame(self) -> Image.Image | None:
        """
        Capture a single frame and return as a PIL Image.

        Returns:
            PIL Image or None if capture fails
        """
        # Grab (decode-skip) stale buffered frames so the next read() is current
        for _ in range(3):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            return None  # Fixed: was missing return

        # Convert from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def release(self):
        """Release the camera resource."""
        if self.cap.isOpened():
            self.cap.release()
            print("📷 Camera released")


# --- Test Block ---
if __name__ == "__main__":
    capture = FrameCapture()
    
    frame = capture.get_frame()
    if frame:
        print(f"✅ Captured frame: {frame.size}")
        frame.save("test_frame.jpg")
        print("💾 Saved to test_frame.jpg")
    
    capture.release()

