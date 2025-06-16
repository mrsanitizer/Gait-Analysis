def analyze_video(video_path):
    import cv2
    import mediapipe as mp
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import csv

    def hip_flexion(hips, knees):
        l1 = hips[1] - knees[1]
        l2 = hips[0] - knees[0]
        l3 = np.sqrt((hips[0] - knees[0])**2 + (hips[1] - knees[1])**2)
        if (l1 == 0 or l2 == l3):
            return 0
        angle = math.atan(l1 / l2) * 180 / math.pi
        return 90- abs(angle)
    
    def knee_flexion(hips, knees, ankles):
        a = np.sqrt((knees[0] - ankles[0])**2 + (knees[1] - ankles[1])**2)
        b = np.sqrt((hips[0] - ankles[0])**2 + (hips[1] - ankles[1])**2)
        c = np.sqrt((knees[0] - hips[0])**2 + (knees[1] - hips[1])**2)
        angle = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
        return abs(angle * 180 / math.pi)
    
    def ankle_up(heels, foot_index):
        if(heels[1] > foot_index[1]):
            return 0
        x = heels[0] - foot_index[0]
        y = heels[1] - foot_index[1]
        angle = math.atan(y/x) * 180 / math.pi
        return abs(angle)

    def ankle_down(heels, foot_index):
        if(heels[1] < foot_index[1]):
            return 0
        x = foot_index[0] - heels[0]
        y = foot_index[1] - heels[1]
        angle = math.atan(y/x) * 180 / math.pi
        return abs(angle)

    # Setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    timestamps = []
    left_hip_flexion_list = []
    right_hip_flexion_list = []
    left_knee_flexion_list = []
    right_knee_flexion_list = []
    left_ankle_up_list = []
    right_ankle_up_list = []
    left_ankle_down_list = []
    right_ankle_down_list = []

    csv_data = []

    # Matplotlib plot (not live)
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    axs[0].set_ylabel('Hip Flexion (deg)')
    axs[1].set_ylabel('Knee Flexion (deg)')
    axs[2].set_ylabel('Ankle Up (deg)')
    axs[3].set_ylabel('Ankle Down (deg)')
    axs[3].set_xlabel('Time (s)')
    axs[0].set_title('Joint Angles Over Time')

    # Annotated video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    annotated_video_path = os.path.join(os.path.dirname(video_path), "annotated_output.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_index / fps, 3)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_coords(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].z])
            def is_visible(idx): return lm[idx].visibility > 0.3

            key_landmarks = [
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                mp_pose.PoseLandmark.LEFT_HEEL,
                mp_pose.PoseLandmark.RIGHT_HEEL
            ]

            if all(is_visible(i) for i in key_landmarks):
                left_hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
                right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
                left_knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE)
                right_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)
                left_ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)
                left_heel = get_coords(mp_pose.PoseLandmark.LEFT_HEEL)
                right_heel = get_coords(mp_pose.PoseLandmark.RIGHT_HEEL)
                left_foot_idx = get_coords(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
                right_foot_idx = get_coords(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

                left_hip_flexion = hip_flexion(left_hip, left_knee)
                right_hip_flexion = hip_flexion(right_hip, right_knee)
                left_knee_flexion = knee_flexion(left_hip, left_knee, left_ankle)
                right_knee_flexion = knee_flexion(right_hip, right_knee, right_ankle)
                left_ankle_up = ankle_up(left_heel, left_foot_idx)
                right_ankle_up = ankle_up(right_heel, right_foot_idx)
                left_ankle_down = ankle_down(left_heel, left_foot_idx)
                right_ankle_down = ankle_down(right_heel, right_foot_idx)

                # Append for plotting
                timestamps.append(timestamp)
                left_hip_flexion_list.append(left_hip_flexion)
                right_hip_flexion_list.append(right_hip_flexion)
                left_knee_flexion_list.append(left_knee_flexion)
                right_knee_flexion_list.append(right_knee_flexion)
                left_ankle_up_list.append(left_ankle_up)
                right_ankle_up_list.append(right_ankle_up)
                left_ankle_down_list.append(left_ankle_down)
                right_ankle_down_list.append(right_ankle_down)

                csv_data.append([
                    timestamp,
                    left_hip_flexion,
                    right_hip_flexion,
                    left_knee_flexion,
                    right_knee_flexion,
                    left_ankle_up,
                    right_ankle_up,
                    left_ankle_down,
                    right_ankle_down
                ])

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Overlay text
                y_offset = 30
                line_height = 30
                for text in [
                    f"L Hip: {left_hip_flexion:.1f}",
                    f"R Hip: {right_hip_flexion:.1f}",
                    f"L Knee: {left_knee_flexion:.1f}",
                    f"R Knee: {right_knee_flexion:.1f}",
                    f"L Ankle Up: {left_ankle_up:.1f}",
                    f"R Ankle Up: {right_ankle_up:.1f}",
                    f"L Ankle Down: {left_ankle_down:.1f}",
                    f"R Ankle Down: {right_ankle_down:.1f}"
                ]:
                    cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    y_offset += line_height

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    pose.close()

    # Save CSV
    csv_output_path = os.path.join(os.path.dirname(video_path), "output_data.csv")
    with open(csv_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "left_hip_flexion", "right_hip_flexion",
            "left_knee_flexion", "right_knee_flexion",
            "left_ankle_up", "right_ankle_up",
            "left_ankle_down", "right_ankle_down"
        ])
        writer.writerows(csv_data)

    # Plot graphs for all metrics
    axs[0].plot(timestamps, left_hip_flexion_list, label="Left Hip", color="blue")
    axs[0].plot(timestamps, right_hip_flexion_list, label="Right Hip", color="red")
    axs[0].legend()

    axs[1].plot(timestamps, left_knee_flexion_list, label="Left Knee", color="blue")
    axs[1].plot(timestamps, right_knee_flexion_list, label="Right Knee", color="red")
    axs[1].legend()

    axs[2].plot(timestamps, left_ankle_up_list, label="Left Ankle Up", color="blue")
    axs[2].plot(timestamps, right_ankle_up_list, label="Right Ankle Up", color="red")
    axs[2].legend()

    axs[3].plot(timestamps, left_ankle_down_list, label="Left Ankle Down", color="blue")
    axs[3].plot(timestamps, right_ankle_down_list, label="Right Ankle Down", color="red")
    axs[3].legend()

    graph_image_path = os.path.join(os.path.dirname(video_path), "graph.png")
    fig.savefig(graph_image_path)
    plt.close(fig)

    result_summary = {
        "avg_left_hip_flexion": round(np.mean(left_hip_flexion_list), 2) if left_hip_flexion_list else 0,
        "avg_right_hip_flexion": round(np.mean(right_hip_flexion_list), 2) if right_hip_flexion_list else 0,
        "avg_left_knee_flexion": round(np.mean(left_knee_flexion_list), 2) if left_knee_flexion_list else 0,
        "avg_right_knee_flexion": round(np.mean(right_knee_flexion_list), 2) if right_knee_flexion_list else 0,
        "avg_left_ankle_up": round(np.mean(left_ankle_up_list), 2) if left_ankle_up_list else 0,
        "avg_right_ankle_up": round(np.mean(right_ankle_up_list), 2) if right_ankle_up_list else 0,
        "avg_left_ankle_down": round(np.mean(left_ankle_down_list), 2) if left_ankle_down_list else 0,
        "avg_right_ankle_down": round(np.mean(right_ankle_down_list), 2) if right_ankle_down_list else 0
    }

    return result_summary, graph_image_path, annotated_video_path, csv_output_path

analyze_video("Your video path here")