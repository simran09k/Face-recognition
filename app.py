import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

st.title("Face Recognition Attendance System")

DATASET_PATH = "dataset"

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance(
name TEXT,
date TEXT,
time TEXT
)
""")

conn.commit()

# -------------------------------
# LOAD DATASET
# -------------------------------
known_faces = []
known_names = []

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_folder):

        for image_name in os.listdir(person_folder):

            image_path = os.path.join(person_folder, image_name)

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.resize(img,(200,200))

            known_faces.append(img)
            known_names.append(person_name)

st.write("Loaded Students:", list(set(known_names)))

# -------------------------------
# FACE DETECTOR
# -------------------------------
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# CAMERA INPUT
# -------------------------------
uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        st.error("No face detected")

    else:

        for (x,y,w,h) in faces:

            face_img = frame[y:y+h,x:x+w]
            face_img = cv2.resize(face_img,(200,200))

            best_score = 999999999
            best_match = None
            best_dataset_image = None

            for i,known_face in enumerate(known_faces):

                difference = np.sum(np.abs(known_face - face_img))

                if difference < best_score:
                    best_score = difference
                    best_match = known_names[i]
                    best_dataset_image = known_face

            THRESHOLD = 45000000

            confidence = max(0,100 - (best_score / THRESHOLD) * 100)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            if best_score < THRESHOLD:

                label = f"{best_match} ({confidence:.2f}%)"

                cv2.putText(frame,label,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                # -------------------------------
                # DUPLICATE CHECK
                # -------------------------------
                cursor.execute("""
                SELECT * FROM attendance
                WHERE name=? AND date=?
                """,(best_match,date))

                result = cursor.fetchone()

                if result:

                    st.warning(f"{best_match} attendance already marked today")

                else:

                    cursor.execute("""
                    INSERT INTO attendance VALUES(?,?,?)
                    """,(best_match,date,time))

                    conn.commit()

                    st.success(f"Attendance marked for {best_match}")

                st.subheader("Match Verification")

                col1,col2 = st.columns(2)

                with col1:
                    st.image(face_img,caption="Captured Face")

                with col2:
                    st.image(best_dataset_image,
                    caption=f"Dataset Image ({best_match})")

            else:

                cv2.putText(frame,"Unknown",(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                st.error("Unknown person")

    st.image(frame,channels="BGR",caption="Detection Result")

# -------------------------------
# SHOW DATABASE RECORDS
# -------------------------------
st.subheader("Attendance Records (Admin View)")

cursor.execute("SELECT * FROM attendance")

rows = cursor.fetchall()

if rows:

    df = pd.DataFrame(rows,columns=["Name","Date","Time"])

    st.dataframe(df)

else:

    st.info("No attendance records yet")
