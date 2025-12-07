import cv2
import mediapipe as mp
import os

# Ask for user's name
name = input("Enter your name: ").strip()

# Create dataset folder
save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

MAX_PHOTOS = 100  # Maximum number of face images to save

mp_face = mp.solutions.face_mesh

# ---------- FIXED CAMERA SELECTION ----------
# Since test showed only camera 0 works, we force it
print("Opening camera 0...")

cap = cv2.VideoCapture(0)   # Linux compatible

if not cap.isOpened():
    print("ERROR: Cannot open /dev/video0!")
    print("Make sure:")
    print(" - Camera is connected")
    print(" - No other app is using it")
    print(" - Permissions allow access")
    exit(1)

print("Camera 0 found and opened successfully.")
# --------------------------------------------

count = 0
reached_limit = False

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as fm:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Draw face rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 2)

                # Crop face region
                face_crop = frame[y_min:y_max, x_min:x_max]
                
                if face_crop.size > 0:
                    cv2.imwrite(f"{save_dir}/{count}.jpg", face_crop)
                    count += 1

                # Stop automatically when enough images are collected
                if count >= MAX_PHOTOS:
                    print(f"Captured {MAX_PHOTOS} images for '{name}'.")
                    reached_limit = True
                    break

        cv2.imshow("Capturing Faces", frame)

        if reached_limit:
            break

        # Press 'q' to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {count} images to: {save_dir}")

