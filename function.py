def analyze_video(video_path):
    """
    Analyzes a video to compute the angles for hip flexion, knee flexion, and ankle dorsiflexion/plantarflexion.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        CSV file and generates plots for the angles.
        Generates an annotated video with the computed angles.
    """
    import cv2
    import mediapipe as mp
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import csv

    def hip_flexion(hips, knees):
        dx = hips[0] - knees[0]
        dy = hips[1] - knees[1]
        if dx == 0:
            return 0
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        # convert to flexion angle
        return 90 - angle

    def knee_flexion(hips, knees, ankles):
        a = np.linalg.norm(knees - ankles)
        b = np.linalg.norm(hips - ankles)
        c = np.linalg.norm(knees - hips)
        if a == 0 or c == 0:
            return 0
        angle = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
        return angle

    def ankle_up(heel, foot_idx):
        # dorsiflexion
        if heel[1] > foot_idx[1]:
            return 0
        dx = heel[0] - foot_idx[0]
        dy = heel[1] - foot_idx[1]
        if dx == 0:
            return 0
        return abs(math.degrees(math.atan2(dy, dx)))

    def ankle_down(heel, foot_idx):
        # plantarflexion
        if heel[1] < foot_idx[1]:
            return 0
        dx = foot_idx[0] - heel[0]
        dy = foot_idx[1] - heel[1]
        if dx == 0:
            return 0
        return abs(math.degrees(math.atan2(dy, dx)))

    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5,
                         model_complexity=0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Data lists
    timestamps = []
    left_hip_list = []
    right_hip_list = []
    left_knee_list = []
    right_knee_list = []
    left_ankle_up_list = []
    right_ankle_up_list = []
    left_ankle_down_list = []
    right_ankle_down_list = []
    csv_data = []

    # Video writer for annotated output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(os.path.dirname(video_path), 'annotated_output.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        res = pose.process(image)
        image.flags.writeable = True

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            def get(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].z])
            vis = lambda idx: lm[idx].visibility > 0.3
            ids = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                   mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                   mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
                   mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            if all(vis(i) for i in ids):
                lh = get(mp_pose.PoseLandmark.LEFT_HIP)
                rh = get(mp_pose.PoseLandmark.RIGHT_HIP)
                lk = get(mp_pose.PoseLandmark.LEFT_KNEE)
                rk = get(mp_pose.PoseLandmark.RIGHT_KNEE)
                lh_ank = get(mp_pose.PoseLandmark.LEFT_HEEL)
                rh_ank = get(mp_pose.PoseLandmark.RIGHT_HEEL)
                lf = get(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
                rf = get(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

                # compute angles
                lhip = hip_flexion(lh, lk)
                rhip = hip_flexion(rh, rk)
                lknee = knee_flexion(lh, lk, lf)
                rknee = knee_flexion(rh, rk, rf)
                lan_up = ankle_up(lh_ank, lf)
                ran_up = ankle_up(rh_ank, rf)
                lan_down = ankle_down(lh_ank, lf)
                ran_down = ankle_down(rh_ank, rf)

                # append for CSV and plotting
                timestamps.append(t)
                left_hip_list.append(lhip)
                right_hip_list.append(rhip)
                left_knee_list.append(lknee)
                right_knee_list.append(rknee)
                left_ankle_up_list.append(lan_up)
                right_ankle_up_list.append(ran_up)
                left_ankle_down_list.append(lan_down)
                right_ankle_down_list.append(ran_down)
                csv_data.append([t, lhip, rhip, lknee, rknee, lan_up, ran_up, lan_down, ran_down])

                # annotate frame
                mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    pose.close()

    # write CSV
    csv_path = os.path.join(os.path.dirname(video_path), 'output_data.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','left_hip','right_hip','left_knee','right_knee',
                    'left_ankle_up','right_ankle_up','left_ankle_down','right_ankle_down'])
        w.writerows(csv_data)

    # plot and save each graph
    metrics = [
        ('left_hip', left_hip_list),
        ('right_hip', right_hip_list),
        ('left_knee', left_knee_list),
        ('right_knee', right_knee_list),
        ('left_ankle_up', left_ankle_up_list),
        ('right_ankle_up', right_ankle_up_list),
        ('left_ankle_down', left_ankle_down_list),
        ('right_ankle_down', right_ankle_down_list)
    ]
    image_paths = {}
    for name, data in metrics:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(timestamps, data, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'{name.replace("_", " ").title()} Over Time')
        ax.legend()
        img_path = os.path.join(os.path.dirname(video_path), f'{name}_graph.png')
        fig.savefig(img_path)
        plt.close(fig)
        image_paths[name] = img_path

    # To get all the plots in a single image
    """
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
    """

    summary = {
        'avg_left_hip': float(np.mean(left_hip_list)),
        'avg_right_hip': float(np.mean(right_hip_list)),
        'avg_left_knee': float(np.mean(left_knee_list)),
        'avg_right_knee': float(np.mean(right_knee_list))
    }
    return summary, image_paths, out_path, csv_path

analyze_video("YOUR_VIDEO_PATH.mp4")