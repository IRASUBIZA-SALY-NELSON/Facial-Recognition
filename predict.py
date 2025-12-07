import cv2
import mediapipe as mp
import json
import platform

# Load LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.xml")

# Load label map
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh

# Detect OS to choose camera backend
def open_camera(index):
    system = platform.system()

    if system == "Windows":
        # Windows uses DirectShow
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        # Linux & macOS use default backend (V4L2 on Linux)
        return cv2.VideoCapture(index)

# Try different camera indices
cap = None
for cam_index in [0, 1, 2]:
    print(f"Trying camera index {cam_index}...")
    test_cap = open_camera(cam_index)

    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None and frame.size > 0:
            if frame.mean() > 5:  # Ensure not all black
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

# Face Mesh processing
with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as fm:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]

                x_min, x_max = max(min(xs), 0), min(max(xs), w)
                y_min, y_max = max(min(ys), 0), min(max(ys), h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 2)

                face_crop = frame[y_min:y_max, x_min:x_max]

                if face_crop.size == 0:
                    cv2.putText(frame, "Face crop empty", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                    continue

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                try:
                    label_id, confidence = recognizer.predict(gray)
                    name = label_map[str(label_id)]
                    text = f"{name} ({int(confidence)})"
                except:
                    text = "Unknown"

                cv2.putText(frame, text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

