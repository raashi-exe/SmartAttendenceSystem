import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 📁 Path
dataset_path = r"C:\Users\lenovo\OneDrive\Documents\SmartAttendanceSystem\imagesclass"
attendance_file = r"C:\Users\lenovo\OneDrive\Documents\SmartAttendanceSystem\attendance.csv"

# 🧠 Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

faces = []
labels = []
label_map = {}
current_label = 0

print("📂 Loading dataset...")

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)

    img = cv2.imread(img_path)

    if img is None:
        print("❌ Failed to load:", file)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(detected_faces) == 0:
        print("❌ No face found in:", file)
        continue

    for (x, y, w, h) in detected_faces:
        faces.append(gray[y:y+h, x:x+w])
        labels.append(current_label)

    name = os.path.splitext(file)[0]
    label_map[current_label] = name
    print(f"✅ Loaded: {name}")

    current_label += 1

# ❌ If no faces found
if len(faces) == 0:
    print("❌ No faces found in any image. Fix your images.")
    exit()

print("✅ Training model...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

print("🎯 Training complete!")

# 📄 CSV setup
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time\n")

cap = cv2.VideoCapture(0)

marked = set()
today_date = datetime.now().strftime("%Y-%m-%d")

print("📷 Camera started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        name = "Unknown"

        if confidence < 80:  # slightly relaxed
            name = label_map[label]

            if name not in marked:
                now = datetime.now()
                time_now = now.strftime("%H:%M:%S")

                df = pd.DataFrame([[name, today_date, time_now]],
                                  columns=["Name", "Date", "Time"])
                df.to_csv(attendance_file, mode='a', header=False, index=False)

                print(f"✅ {name} marked present")
                marked.add(name)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
