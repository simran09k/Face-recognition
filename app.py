import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

st.title("Face Recognition Attendance System")

DATASET_PATH = "dataset"

known_faces = []
known_names = []

# Load dataset
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img,(200,200))
        known_faces.append(img)
        known_names.append(person_name)

st.write("Loaded Students:", list(set(known_names)))

# Upload photo
uploaded_file = st.camera_input("Take a picture")

attendance = []

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    captured_img = cv2.imdecode(file_bytes, 1)
    captured_img = cv2.resize(captured_img,(200,200))

    st.image(captured_img, caption="Captured Image")

    best_score = 999999999
    best_match = None
    best_dataset_image = None

    # Compare with dataset
    for i, face in enumerate(known_faces):

        difference = np.sum(np.abs(face - captured_img))

        if difference < best_score:
            best_score = difference
            best_match = known_names[i]
            best_dataset_image = face

    THRESHOLD = 45000000

    st.write("Similarity Score:", best_score)

    if best_score < THRESHOLD:

        st.success(f"Attendance marked for {best_match}")

        now = datetime.now()
        attendance.append([best_match, now.strftime("%H:%M:%S")])

        st.subheader("Match Verification")

        col1, col2 = st.columns(2)

        with col1:
            st.image(captured_img, caption="Captured Image")

        with col2:
            st.image(best_dataset_image, caption=f"Matched Dataset Image ({best_match})")

    else:
        st.error("Unknown Person - Not in Dataset")

# Save attendance
if attendance:
    df = pd.DataFrame(attendance, columns=["Name","Time"])
    df.to_csv("attendance.csv", index=False)

    st.download_button(
        "Download Attendance",
        df.to_csv(index=False),
        "attendance.csv"
    )
