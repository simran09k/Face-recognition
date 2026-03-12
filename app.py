import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

st.title("Face Recognition Attendance System")

DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"

known_faces = []
known_names = []

# Load dataset
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):

            image_path = os.path.join(person_folder, image_name)

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.resize(img, (200,200))

            known_faces.append(img)
            known_names.append(person_name)

st.write("Loaded Students:", list(set(known_names)))

uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    captured_img = cv2.imdecode(file_bytes, 1)
    captured_img = cv2.resize(captured_img,(200,200))

    st.image(captured_img, caption="Captured Image")

    best_score = 999999999
    best_match = None
    best_dataset_image = None

    for i, face in enumerate(known_faces):

        difference = np.sum(np.abs(face - captured_img))

        if difference < best_score:
            best_score = difference
            best_match = known_names[i]
            best_dataset_image = face

    THRESHOLD = 45000000

    st.write("Similarity Score:", best_score)

    if best_score < THRESHOLD:

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        # Check if attendance file exists
        if os.path.exists(ATTENDANCE_FILE):

            df = pd.read_csv(ATTENDANCE_FILE)

            duplicate = ((df["Name"] == best_match) & (df["Date"] == date)).any()

            if duplicate:
                st.warning(f"{best_match} attendance already marked today")

            else:
                new_entry = pd.DataFrame([[best_match,date,time]],columns=["Name","Date","Time"])
                new_entry.to_csv(ATTENDANCE_FILE,mode='a',header=False,index=False)
                st.success(f"Attendance marked for {best_match}")

        else:

            df = pd.DataFrame([[best_match,date,time]],columns=["Name","Date","Time"])
            df.to_csv(ATTENDANCE_FILE,index=False)

            st.success(f"Attendance marked for {best_match}")

        st.subheader("Match Verification")

        col1, col2 = st.columns(2)

        with col1:
            st.image(captured_img, caption="Captured Image")

        with col2:
            st.image(best_dataset_image, caption=f"Matched Dataset Image ({best_match})")

    else:
        st.error("Unknown Person - Not in Dataset")


# Show attendance table
if os.path.exists(ATTENDANCE_FILE):

    st.subheader("Attendance Records")

    df = pd.read_csv(ATTENDANCE_FILE)

    st.dataframe(df)

    st.download_button(
        label="Download Attendance CSV",
        data=df.to_csv(index=False),
        file_name="attendance.csv",
        mime="text/csv"
    )
