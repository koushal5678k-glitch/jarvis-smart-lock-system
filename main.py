import cv2
import os
import time

from voice_module import VoiceModule
from vision_module import VisionModule
from recognition_module import RecognitionModule

class JarvisAssistant:
    def __init__(self):
        self.voice = VoiceModule()
        self.vision = VisionModule(camera_index=0)
        self.recognition = RecognitionModule(known_faces_dir="known_faces")
        
        # Load user encodings at startup
        self.voice.speak("Initializing Jarvis systems. Loading known faces.")
        self.recognition.load_known_faces()
        
        self.greeting = self.voice.get_time_greeting()
        self.voice.speak(f"{self.greeting}! I am online and ready.")

    def run_idle_loop(self):
        """State 0: Listen for wake word."""
        while True:
            is_awake = self.voice.listen_for_wake_word()
            if is_awake:
                self.voice.speak_async("Yes, I am listening.")
                self.process_command()

    def process_command(self):
        """State 1: Listen for explicit command and transition."""
        command = self.voice.listen_for_command()
        
        if command:
            if "open camera" in command:
                self.voice.speak_async("Opening camera.")
                self.run_camera_mode(detect_faces=False)
            elif "start detection" in command:
                self.voice.speak_async("Starting face detection and security mode.")
                self.run_camera_mode(detect_faces=True)
            elif "who am i" in command:
                self.voice.speak_async("Let me check.")
                self.identify_user_once()
            elif "exit" in command or "stop" in command:
                self.voice.speak("Shutting down the system. Goodbye!")
                exit(0)
            else:
                self.voice.speak_async("I didn't catch a valid command. Returning to standby.")
        else:
            self.voice.speak_async("No command received. Returning to standby.")

    def run_camera_mode(self, detect_faces=False):
        """State 2: Active Camera Loop."""
        self.vision.start_camera()
        print("Press 'q' in the video window to return to standby.")
        
        greeted_users = set()  # To avoid spamming greetings
        
        while True:
            frame = self.vision.read_frame()
            if frame is None:
                print("Failed to grab frame.")
                break

            # Smart add-on: Image quality
            quality = self.vision.assess_image_quality(frame)
            if not quality["is_bright"]:
                cv2.putText(frame, "Low Light", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not quality["is_clear"]:
                cv2.putText(frame, "Blurry", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if detect_faces:
                # Recognize and detect faces
                faces = self.recognition.recognize_faces(frame)
                
                for face in faces:
                    name = face["name"]
                    loc = face["location"]
                    conf = face["confidence"]
                    
                    # Smart Add-on: Check occlusion hint
                    # (In this simple implementation, if face detection fails occasionally we assume occlusion)
                    # For a specific hint we could use the face_region logic, but face_recognition natively drops it if occluded.
                    
                    if name == "Unknown":
                        color = (0, 0, 255) # Red for unknown
                    else:
                        color = (0, 255, 0) # Green for known
                        # Time-based smart greeting for recognized user
                        if name not in greeted_users:
                            self.voice.speak_async(f"Hello {name}, welcome back!")
                            greeted_users.add(name)
                            
                    # Draw box
                    self.vision.draw_face_box(frame, loc, name, conf, color=color)

                # Smart Add-on: Security warning
                if self.recognition.security_warning_triggered:
                    self.voice.speak_async("Warning! Unknown person detected.")
                    self.recognition.reset_security_warning()  # Reset so it doesn't spam every frame (will trigger again if they stay)
                    time.sleep(1) # Prevent immediate re-trigger spamming

            self.vision.display_frame(frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.voice.speak_async("Closing camera.")
                break
                
        self.vision.stop_camera()

    def identify_user_once(self):
        """Opens camera briefly to identify the user."""
        self.vision.start_camera()
        
        # Discard a few frames for camera warmup
        for _ in range(5):
            self.vision.read_frame()
            time.sleep(0.1)
            
        frame = self.vision.read_frame()
        self.vision.stop_camera()
        
        if frame is not None:
            # Smart capture logic: check quality
            quality = self.vision.assess_image_quality(frame)
            if not quality["is_clear"] or not quality["is_bright"]:
                self.voice.speak("I cannot see you clearly. Please ensure good lighting and look directly at the camera.")
                return

            faces = self.recognition.recognize_faces(frame)
            if faces:
                # Sort by size to get largest face
                # loc format is (top, right, bottom, left)
                largest_face = sorted(faces, key=lambda f: (f["location"][2] - f["location"][0]) * (f["location"][1] - f["location"][3]), reverse=True)[0]
                
                if largest_face["name"] != "Unknown":
                    self.voice.speak(f"You are {largest_face['name']}.")
                else:
                    self.voice.speak("I don't recognize you.")
            else:
                self.voice.speak("I could not detect any faces.")
        else:
            self.voice.speak("Camera capture failed.")

if __name__ == "__main__":
    try:
        assistant = JarvisAssistant()
        assistant.run_idle_loop()
    except KeyboardInterrupt:
        print("\nShutting down Jarvis.")
