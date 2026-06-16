import cv2
import numpy as np

class VisionModule:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self):
        """Open the webcam."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            # You can set resolution here if needed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_frame(self):
        """Read a single frame from the webcam."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def stop_camera(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def assess_image_quality(self, frame):
        """
        Smart Add-on: Returns dict of quality metrics.
        - Checks brightness via mean pixel intensity.
        - Checks blurriness via variance of the Laplacian.
        """
        if frame is None:
            return {"is_clear": False, "is_bright": False, "blur_score": 0, "brightness": 0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur check
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_clear = blur_score > 100.0  # Threshold for blur (can be adjusted)
        
        # Brightness check
        brightness_score = np.mean(gray)
        is_bright = brightness_score > 50.0 # Threshold for low light
        
        return {
            "is_clear": is_clear, 
            "is_bright": is_bright, 
            "blur_score": blur_score, 
            "brightness": brightness_score
        }

    def check_occlusion(self, face_region):
        """
        Smart Add-on: Basic occlusion/emotion hint based on face region visibility.
        If the face bounding box is too small or missing features, it might be occluded.
        (A simplified proxy for "I cannot see you clearly")
        """
        # A simple check: if the face width/height is less than expected, or if we can't extract it.
        if face_region is None or face_region.size == 0:
            return True # Occluded
        return False

    def draw_face_box(self, frame, location, label, confidence, color=(0, 255, 0)):
        """Draw bounding box and text on the frame."""
        # Location format expected: (top, right, bottom, left) natively from face_recognition
        # Or (x,y,w,h) from opencv. Let's stick to (top, right, bottom, left) convention
        top, right, bottom, left = location

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Draw labels
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{label} ({confidence:.2f})" if confidence else label
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    def display_frame(self, frame, window_name="Jarvis UI"):
        """Display the frame in an OpenCV window."""
        cv2.imshow(window_name, frame)
