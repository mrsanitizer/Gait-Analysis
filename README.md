# Readme

This script analyzes a video of human movement using MediaPipe Pose and OpenCV, extracts joint angles and gait metrics, and produces:
- An annotated video with angle overlays
- A CSV file with all computed metrics per frame
- A multi-panel graph image showing the time series of all joint angles

USAGE:
------
1. Place your input video (e.g., Right_neptune.mp4) in the same directory as Combined_graph.py.
2. Run the script:
   python3 Combined_graph.py

WHAT THE SCRIPT DOES:
---------------------
- Detects pose landmarks for each frame of the video.
- Calculates:
    * Hip flexion (left/right)
    * Knee flexion (left/right)
    * Ankle up angle (left/right)
    * Ankle down angle (left/right)
    * Gait metrics: foot_ahead (which foot is ahead), straight_walk (is pelvis aligned with feet)
- Saves all metrics to output_data.csv with columns:
    timestamp, left_hip_flexion, right_hip_flexion, left_knee_flexion, right_knee_flexion, left_ankle_up, right_ankle_up, left_ankle_down, right_ankle_down, foot_ahead, straight_walk
- Draws all angles and gait metrics as text overlays on the output video (annotated_output.mp4).
- Plots all joint angles over time in a 4-panel graph (graph.png).

OUTPUT FILES:
-------------
- output_data.csv: All metrics per frame.
- annotated_output.mp4: Video with angle overlays.
- graph.png: Multi-panel plot of all joint angles.

DEPENDENCIES:
-------------
- Python 3.x
- OpenCV (cv2)
- mediapipe
- numpy
- matplotlib

INSTALLATION
------------
To install all required Python packages, run:

```pip install opencv-python mediapipe numpy matplotlib```

Or, if you have a requirements.txt file, run:

```pip install -r requirements.txt```

TROUBLESHOOTING:
----------------
- If you get errors about missing modules, install them with pip (e.g., pip install opencv-python mediapipe numpy matplotlib).
- If the output video is rotated, check your input video orientation or use a video player that respects orientation metadata.
