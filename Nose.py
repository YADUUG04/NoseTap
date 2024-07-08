import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Hiding Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Initialize Face Detection and Hands modules
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def main():
    st.title("Noise Tap Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture("uploaded_video.mp4")

        lap_timers = []
        lap_counts = 0
        lap_start_time = None

        smooth_factor = 0.2
        prev_ix, prev_iy, prev_nx, prev_ny = 0, 0, 0, 0

        cooldown_period = 1.0
        last_lap_time = None

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Error reading frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            ix, iy, nx, ny = prev_ix, prev_iy, prev_nx, prev_ny

            if hands_results.multi_hand_landmarks:
                landmarks = hands_results.multi_hand_landmarks[0]
                index_fingertip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                height, width, _ = frame.shape
                ix, iy = int(index_fingertip.x * width), int(index_fingertip.y * height)

                if lap_start_time is None:
                    lap_start_time = time.time()

                if face_results.detections:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        nx = int((bboxC.xmin + bboxC.width / 2) * iw)
                        ny = int((bboxC.ymin + bboxC.height / 2) * ih)

                        ix = int((1 - smooth_factor) * prev_ix + smooth_factor * ix)
                        iy = int((1 - smooth_factor) * prev_iy + smooth_factor * iy)
                        nx = int((1 - smooth_factor) * prev_nx + smooth_factor * nx)
                        ny = int((1 - smooth_factor) * prev_ny + smooth_factor * ny)

                        cv2.line(frame, (nx, ny), (ix, iy), (0, 255, 0), 2)

                        if ((ix - nx) ** 2 + (iy - ny) ** 2) ** 0.5 < 30:
                            lap_end_time = time.time()
                            if last_lap_time is None or (lap_end_time - last_lap_time) > cooldown_period:
                                lap_timer = lap_end_time - lap_start_time
                                if lap_timer > 0.2:
                                    lap_counts += 1
                                    lap_timers.append(lap_timer)
                                    lap_start_time = lap_end_time
                                    last_lap_time = lap_end_time

                        prev_ix, prev_iy, prev_nx, prev_ny = ix, iy, nx, ny

            if hands_results.multi_hand_landmarks:
                cv2.circle(frame, (ix, iy), 10, (255, 0, 0), cv2.FILLED)

            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    nx = int((bboxC.xmin + bboxC.width / 2) * iw)
                    ny = int((bboxC.ymin + bboxC.height / 2) * ih)
                    cv2.circle(frame, (nx, ny), 10, (0, 255, 0), cv2.FILLED)

            if lap_timers:
                lap_timer = lap_timers[-1]
                average_speed = np.mean(lap_timers) if lap_timers else 0

                cv2.putText(frame, f"Lap Timer: {lap_timer:.2f} seconds", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Lap Count: {lap_counts}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Average Speed: {average_speed:.2f} s/lap", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)

        cap.release()

        if lap_timers:
            csv_data = "Lap Number,Lap Time (seconds)\n"
            for i, lap_time in enumerate(lap_timers, start=1):
                csv_data += f"{i},{lap_time}\n"

            st.download_button(
                label="Download Lap Times CSV",
                data=csv_data.encode(),
                file_name="lap_times.csv",
                mime="text/csv"
            )

        average_speed = np.mean(lap_timers) if lap_timers else 0
        st.write(f"Average Lap Timer: {average_speed:.2f} seconds per lap")

        if lap_timers:
            plt.plot(range(1, len(lap_timers) + 1), lap_timers, marker='o')
            plt.xlabel('Lap Number')
            plt.ylabel('Lap Time (seconds)')
            plt.title('Lap Time Progression')
            st.pyplot(plt)

if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if st.session_state['authenticated']:
        main()
    else:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state['authenticated'] = True
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
