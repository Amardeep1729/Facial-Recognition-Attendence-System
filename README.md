# Face Attendance System

This is a simple face recognition-based attendance system using Python, Mediapipe, and DeepFace.

---

## Features

- Detects faces using Mediapipe
- Recognizes faces using DeepFace with Facenet model
- Marks attendance with name and timestamp in a CSV file named with the current date
- Shows cropped face in a mini window and full webcam feed with attendance info
- Allows adding new faces by pressing `S` and entering the name
- Avoids duplicate attendance entries

---

## How to Use

1. **Install dependencies:**

   ```bash
   pip install opencv-python mediapipe deepface numpy
   ```

2. **Create a folder named `faces`** in the project directory and add known face images there (image name should be the person's name).

3. **Run the script:**

   ```bash
   python face_attendance.py
   ```

4. The system will open your webcam, detect and recognize faces, and mark attendance in a CSV file with the current date.

5. If a face is not recognized, press **`S`** to save a new face with a name.

6. Press **`Q`** to quit.

---

## Output

- Attendance is saved in a CSV file named like `DD-MM-YYYY.csv`
- The CSV file contains two columns: Name and Time

---

## Notes

- Make sure the lighting is good for better face detection.
- Webcam must be accessible.
- The system matches faces with images in the `faces` folder.
- CSV file appends new entries, avoids duplicates.

---

## License

This project is for educational use.

---
