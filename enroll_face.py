import cv2
import os

def enroll_new_user():
    print("=== Jarvis System: Enroll New Face ===")
    name = input("Enter the name of the person: ").strip()
    
    if not name:
        print("Name cannot be empty. Exiting.")
        return

    cam = cv2.VideoCapture(0)
    # Give the camera a second to warm up
    import time
    time.sleep(1)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # We will create a dedicated folder inside known_faces for this user
    save_dir = os.path.join("known_faces", name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\nLook at the camera! Moving your head slightly will improve accuracy.")
    print(f"I will now automatically capture 50 photos for {name}...")
    
    count = 0
    while count < 50:
        ret, frame = cam.read()
        if not ret: 
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            # We save only the cropped face in grayscale to perfectly train LBPH
            face_roi = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_dir}/{count}.jpg", face_roi)
            
            # Draw UI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Give a small delay so photos capture slightly different micro-expressions
            cv2.waitKey(100) 
            break # only capture one face per frame
            
        cv2.imshow("Jarvis Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled early by user.")
            break
            
    cam.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Captured {count} photos for {name}.")
    print("You can now run main.py and Jarvis will recognize you perfectly!")
    
if __name__ == "__main__":
    enroll_new_user()
