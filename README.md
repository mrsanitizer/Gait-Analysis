# NeuroVision: Lower Body Kinematics

NeuroVision is a computer vision tool for analyzing human movement. This module extracts and processes lower-body joint angles from standard video using MediaPipe pose estimation.

## Features

* **Kinematic Tracking:** Calculates bilateral hip flexion, knee flexion, dorsiflexion, and plantarflexion over time.
* **Cross-Platform GUI:** Built-in graphical interface for video processing and native annotated playback.
* **Data Export:** Automatically generates time-series plots (`.png`) and raw frame-by-frame data (`.csv`).
* **Deterministic Environment:** Uses `uv` and strict dependency locking for guaranteed reproducibility across Linux, macOS, and Windows.

## Setup

This project uses `uv` for rapid, reproducible environment management.

### 1. Clone the repository

```bash
git clone https://github.com/mrsanitizer/Gait-Analysis.git
cd Gait-Analysis
```

### 2. Create the virtual environment

This project strictly requires Python 3.11, which is automatically enforced by the `.python-version` file.

```bash
uv venv
```

### 3. Activate the environment

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install the locked dependencies

```bash
uv pip install -r requirements.lock
```

## Usage

Launch the graphical interface:

```bash
python gui.py
```
Select the "example_vid.mp4" file, wait for few seconds! The output will be saved in the folder <output_filename>
