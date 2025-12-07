import cv2
import mediapipe as mp
import os

name = input("Enter your name: ").strip()

save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

# Maximum number of photos to capture
MAX_PHOTOS = 100

mp_face = mp.solutions.face_mesh

# Try different camera indices to find a working camera
cap = None
for cam_index in [0, 1, 2]:
    print(f"Trying camera index {cam_index}...")
    # Use DirectShow backend on Windows for better compatibility
    test_cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None and frame.size > 0:
            # Check if frame is not all black
            if frame.mean() > 5:  # Average pixel value > 5 means not black
                cap = test_cap
                print(f"Camera found at index {cam_index}")
                break
            else:
                print(f"Camera {cam_index} returns black frames, trying next...")
        else:
            print(f"Camera {cam_index} cannot read frames, trying next...")
        test_cap.release()
    else:
        print(f"Camera {cam_index} not available, trying next...")

if cap is None:
    print("ERROR: No working camera found!")
    print("Please check:")
    print("  1. Is your webcam connected?")
    print("  2. Is another app using the camera?")
    print("  3. Are camera permissions granted?")
    exit(1)

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

                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 2)

                # Crop & save
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size > 0:
                    cv2.imwrite(f"{save_dir}/{count}.jpg", face_crop)
                    count += 1

                    # Stop when we have enough photos
                    if count >= MAX_PHOTOS:
                        print(f"Reached {MAX_PHOTOS} photos for '{name}'. Stopping capture.")
                        reached_limit = True
                        break

        cv2.imshow("Capturing Faces", frame)

        # If we've reached the limit inside the face loop, break the main loop
        if reached_limit:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {count} images to {save_dir}")