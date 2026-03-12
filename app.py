import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image

st.title("Face Recognition Attendance System")

dataset_path = "dataset"

# Load dataset images
known_faces = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)
        image = cv2.imread(img_path)

        if image is not None:
            image = cv2.resize(image, (200, 200))
            known_faces.append(image)
            known_names.append(person_name)

# Create attendance file if not exists
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv("attendance.csv", index=False)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.header("Capture Image")

uploaded_file = st.camera_input("Take a picture")

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    df = pd.read_csv("attendance.csv")

    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        df.loc[len(df)] = [name, date, time]
        df.to_csv("attendance.csv", index=False)
        st.success(f"Attendance marked for {name}")
    else:
        st.warning(f"{name} already marked today")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:

        face_img = image[faces[0][1]:faces[0][1]+faces[0][3],
                         faces[0][0]:faces[0][0]+faces[0][2]]

        face_img = cv2.resize(face_img, (200,200))

        best_match = None
        best_score = 999999

        for i, known_face in enumerate(known_faces):

            diff = np.sum((known_face.astype("float") - face_img.astype("float")) ** 2)

            if diff < best_score:
                best_score = diff
                best_match = known_names[i]

        if best_score < 50000000:
            mark_attendance(best_match)
        else:
            st.error("Face not recognized")

    else:
        st.error("No face detected")

st.header("Download Attendance")

with open("attendance.csv", "rb") as file:
    st.download_button(
        label="Download attendance.csv",
        data=file,
        file_name="attendance.csv",
        mime="text/csv"
    )