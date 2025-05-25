import cv2
import mediapipe as mp
from deepface import DeepFace
import os
import numpy as np
from datetime import datetime
import csv

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Function to load known faces from folder
def load_known_faces():
    known = {}
    for filename in os.listdir("faces"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            known[name] = os.path.join("faces", filename)
    return known

known_faces = load_known_faces()
marked_present = set()

# Attendance log
now = datetime.now()
date_str = now.strftime("%d-%m-%Y")
csv_path = f"{date_str}.csv"

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    
    if results.detections:
        # Sort and take the largest face (most prominent one)
        detections = sorted(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)
        detection = detections[0]  # Pick the first (largest) detection

        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Crop the face region
        face = frame[y:y + height, x:x + width]
        if face.size == 0:
            continue

        # Resize and save temporarily
        face_resized = cv2.resize(face, (224, 224))
        temp_image_path = "temp_face.jpg"
        cv2.imwrite(temp_image_path, face_resized)

        # Debug window for cropped face
        cv2.imshow("Cropped Face", face_resized)

        recognized = False
        for name, image_path in known_faces.items():
            if name in marked_present:
                continue  # Skip already marked people

            try:
                result = DeepFace.verify(temp_image_path, image_path, enforce_detection=False, model_name="Facenet")
                print(f"Checking {name} - Verified: {result['verified']}, Distance: {result['distance']:.4f}")

                if result["verified"]:
                    recognized = True
                    time_str = datetime.now().strftime("%H:%M:%S")
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, time_str])
                    marked_present.add(name)
                    print(f"[INFO] Marked {name} present at {time_str}")
                    cv2.putText(frame, f"{name} Present", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break

            except Exception as e:
                print(f"Error verifying {name}: {e}")

        if not recognized:
            cv2.putText(frame, "Unknown - Press S to Save", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                name = input("Enter name for this new face: ")
                filename = os.path.join("faces", f"{name}.jpg")
                cv2.imwrite(filename, face_resized)
                known_faces = load_known_faces()
                print(f"[INFO] {name}'s face saved.")

    # Show the full video frame
    cv2.putText(frame, f"Total Present: {len(marked_present)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Face Attendance", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
f.close()
