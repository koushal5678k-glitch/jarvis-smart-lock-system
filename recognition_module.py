import cv2
import os
import numpy as np

class RecognitionModule:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        # OpenCV built-in face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # OpenCV built-in face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.known_face_names = {} # Map integer ID to name
        self.is_trained = False
        
        # Security state variables
        self.unknown_person_count = 0
        self.security_warning_triggered = False

    def load_known_faces(self):
        """Detect faces in known_faces directory and train the LBPH Recognizer."""
        print(f"Loading known faces from {self.known_faces_dir}...")
        
        if not os.path.exists(self.known_faces_dir):
            print(f"Directory {self.known_faces_dir} not found. Creating it.")
            os.makedirs(self.known_faces_dir)
            return

        faces = []
        ids = []
        
        # We will map each unique name to a specific ID
        name_to_id = {}
        current_new_id = 0
        
        valid_ext = ('.jpg', '.jpeg', '.png')
        
        # os.walk allows checking both root folder and subdirectories
        for root, dirs, files in os.walk(self.known_faces_dir):
            for filename in files:
                if filename.lower().endswith(valid_ext):
                    filepath = os.path.join(root, filename)
                    
                    # If it's in a subdirectory, use the folder name. Else use filename.
                    folder_name = os.path.basename(root)
                    if folder_name == os.path.basename(self.known_faces_dir):
                        name = os.path.splitext(filename)[0]
                    else:
                        name = folder_name
                        
                    # Assign a permanent integer ID to this name
                    if name not in name_to_id:
                        name_to_id[name] = current_new_id
                        self.known_face_names[current_new_id] = name
                        current_new_id += 1
                        
                    try:
                        img = cv2.imread(filepath)
                        if img is None: continue
                            
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Detect face to crop it
                        detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        # Sometimes the image is ALREADY cropped (if gathered by enroll_face.py)
                        if len(detected_faces) == 0:
                            # Assume the whole image is the face if it's relatively square
                            faces.append(gray)
                            ids.append(name_to_id[name])
                        else:
                            (x, y, w, h) = detected_faces[0]
                            face_roi = gray[y:y+h, x:x+w]
                            faces.append(face_roi)
                            ids.append(name_to_id[name])
                            
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        
        # Train recognizer if we have at least one face
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.is_trained = True
            unique_names = list(set(self.known_face_names.values()))
            print(f"Successfully trained the LBPH Face Recognizer on {len(faces)} photos for {len(unique_names)} people: {unique_names}")
        else:
            print("No valid faces were loaded. Recognition disabled.")

    def recognize_faces(self, frame):
        """
        Detect and recognize faces in a BGR frame using OpenCV.
        Returns a list of dictionaries containing name, location, and confidence data.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        unknown_detected = False

        for (x, y, w, h) in detected_faces:
            name = "Unknown"
            confidence_score = 0.0
            
            if self.is_trained:
                # Crop and predict
                face_roi = gray[y:y+h, x:x+w]
                id_, confidence = self.recognizer.predict(face_roi)
                
                # In LBPH, lower confidence is better (it represents distance).
                threshold = 85
                if confidence < threshold:
                    name = self.known_face_names.get(id_, "Unknown")
                    # Convert distance to a 0.0-1.0 confidence score for the UI
                    confidence_score = max(0.0, 1.0 - (confidence / 150.0))
                else:
                    unknown_detected = True
            else:
                unknown_detected = True
                
            # Convert typical OpenCV (x,y,w,h) to face_recognition style (top, right, bottom, left)
            top, right, bottom, left = y, x + w, y + h, x
            
            results.append({
                "name": name,
                "location": (top, right, bottom, left),
                "confidence": confidence_score
            })

        # Smart Security Add-on logic
        if unknown_detected:
            self.unknown_person_count += 1
            if self.unknown_person_count > 10 and not self.security_warning_triggered:
                self.security_warning_triggered = True
        else:
            self.unknown_person_count = 0
            self.security_warning_triggered = False

        return results

    def reset_security_warning(self):
        """Reset the warning flag after it has been handled."""
        self.security_warning_triggered = False
        self.unknown_person_count = 0
