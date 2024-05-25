import cv2
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt
import streamlit as st

# Hiding
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
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Define Streamlit app
def main():
    st.title("Hand and Face Detection")

    # Get user input for video file upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        # Open a connection to the uploaded video file
        cap = cv2.VideoCapture("uploaded_video.mp4")

        lap_timers = []
        lap_counts = 0
        lap_start_time = None

        # Initialize variables for smoothing
        smooth_factor = 0.5
        prev_ix, prev_iy, prev_nx, prev_ny = 0, 0, 0, 0

        csv_file_path = 'lap_times.csv'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Error reading frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb_frame)
            hands_results = hands.process(rgb_frame)

            if hands_results.multi_hand_landmarks:
                # Find the hand with the highest confidence (first in the list)
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

                        # Smooth out the movement
                        ix = int((1 - smooth_factor) * prev_ix + smooth_factor * ix)
                        iy = int((1 - smooth_factor) * prev_iy + smooth_factor * iy)
                        nx = int((1 - smooth_factor) * prev_nx + smooth_factor * nx)
                        ny = int((1 - smooth_factor) * prev_ny + smooth_factor * ny)

                        # Draw line connecting nose and index fingertip
                        cv2.line(frame, (nx, ny), (ix, iy), (0, 255, 0), 2)

                        # Check for lap completion
                        if ((ix - nx) ** 2 + (iy - ny) ** 2) ** 0.5 < 30:
                            lap_end_time = time.time()
                            lap_timer = lap_end_time - lap_start_time
                            if lap_timer > 1:  # Ignore very short durations
                                lap_counts += 1
                                lap_timers.append(lap_timer)
                                lap_start_time = lap_end_time

                        # Save current coordinates for smoothing
                        prev_ix, prev_iy, prev_nx, prev_ny = ix, iy, nx, ny

            # Draw circle on index fingertip
            if hands_results.multi_hand_landmarks:
                cv2.circle(frame, (ix, iy), 10, (255, 0, 0), cv2.FILLED)

            # Draw circle on nose
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    nx = int((bboxC.xmin + bboxC.width / 2) * iw)
                    ny = int((bboxC.ymin + bboxC.height / 2) * ih)
                    cv2.circle(frame, (nx, ny), 10, (0, 255, 0), cv2.FILLED)

            # Display lap timer in seconds, lap count, and average speed
            if lap_timers:
                lap_timer = lap_timers[-1]

                cv2.putText(frame, f"Lap Timer: {lap_timer:.2f} seconds", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Lap Count: {lap_counts}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert the frame to RGB for displaying with Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Download button for CSV file
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

        # Release the video capture object
        cap.release()

        # Calculate average lap time
        average_speed = sum(lap_timers) / len(lap_timers) if len(lap_timers) > 0 else 0
        st.write(f"Average Lap Timer: {average_speed:.2f} seconds per lap")

        # Plot the lap time progression
        if lap_timers:
            plt.plot(range(1, len(lap_timers) + 1), lap_timers, marker='o')
            plt.xlabel('Lap Number')
            plt.ylabel('Lap Time (seconds)')
            plt.title('Lap Time Progression')
            st.pyplot(plt)

# Run the Streamlit app
if __name__ == "__main__":
    main()

