# 🛠️ RoboSahaya — Setup and Verification Guide

This guide provides end-to-end instructions for running and verifying **RoboSahaya** on your machine.

---

## 💻 System Requirements

| Mode | Supported Operating Systems | Requirements |
|---|---|---|
| **Fixture Replay Mode (`--mode mock`)** | Ubuntu, Debian, macOS, Windows 10/11 | Python 3.11 or 3.12; studio runtime/build is unavailable |
| **Live ROS 2 Mode (`--mode live`)** | Ubuntu 24.04 LTS (Noble Numbat) | Python 3.12, ROS 2 Jazzy, DDS middleware (FastDDS / CycloneDDS) |

---

## ⚡ 1. Rapid Installation (Under 60 Seconds)

### Step 1: Navigate to Workspace
```bash
cd /media/sunraku/workspace/gnani_ai
```

### Step 2: Initialize Virtual Environment
Using `uv` (recommended for speed) or standard Python `venv`:
```bash
# Using standard venv with system packages (allows rclpy inheritance on ROS 2 hosts)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 2. Environment Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```

Edit `.env` if you have a Gnani AI API key:
```ini
# Gnani API credentials, if available
GNANI_STT_API_KEY_ID=
GNANI_TTS_API_KEY=
GNANI_STT_URL=https://api.vachana.ai/stt/v3
GNANI_TTS_URL=https://api.gnani.ai/v1/tts/inference
GNANI_TTS_AUTH_SCHEME=Bearer

# Optional reasoning endpoint
EVON_API_URL=
EVON_API_KEY=
EVON_MODEL_ID=evon

# Execution Mode ('mock' for evaluator replay, 'live' for real robot)
ROBOSAHAYA_MODE=mock
ROBOSAHAYA_MOCK_SCENARIO=qos_reliability_mismatch

# ROS Domain
ROS_DOMAIN_ID=0
```

> Without provider credentials, STT and TTS are unavailable; typed input and deterministic fixture diagnostics continue to work. No speech output is simulated.

---

## 🚀 3. Launching the Application

### Option A: Mock Replay Mode (Recommended for Judges)
Runs with zero external robotics hardware or ROS 2 dependencies. Replays exact ROS 2 graph snapshots:
```bash
PYTHONPATH=src python -m robosahaya.app --mode mock
```
Open **`http://127.0.0.1:7860`** in your browser.

Choose a fixture scenario at startup. These are replay diagnoses, not live ROS observations:
1. `qos_reliability_mismatch`: LiDAR publisher `BEST_EFFORT` vs Subscriber `RELIABLE`
2. `missing_laser_pub`: Topic `/scan` exists, but 0 publishers active
3. `tf_broken_camera_link`: Disconnected `base_link` $\rightarrow$ `camera_link` coordinate frame
4. `sim_time_divergence`: Paused `/clock` vs `use_sim_time: false` on node
5. `clean_healthy_system`: Fully operational robot

For example: `PYTHONPATH=src python -m robosahaya.app --mode mock --scenario qos_reliability_mismatch`.

### Option B: Live ROS 2 Mode (Real Hardware or Gazebo)
To inspect an active physical robot or Gazebo simulation running on your machine:
```bash
# Source ROS 2 underlay
source /opt/ros/jazzy/setup.bash

# Ensure same domain ID as your robot
export ROS_DOMAIN_ID=0

# Launch RoboSahaya in live mode
PYTHONPATH=src python -m robosahaya.app --mode live
```

---

## 🧪 4. Automated Verification & Benchmarking

### Run the Pytest Suite
```bash
pytest tests/ -v
```
Results vary by revision and environment; no fixed pass count or runtime is asserted here.

### Run the Performance Benchmark
```bash
PYTHONPATH=src python evaluation/run_benchmark.py
```
This computes diagnostic and grounding measurements from the fixture set. The separate studio runtime benchmark is reported as not run; it does not claim live Gazebo or sensor results.

---

## 🔍 5. Troubleshooting & FAQ

1. **"Gradio web page is blank or not opening"**  
   Ensure port 7860 is not in use by another service. You can specify a custom port:
   ```bash
   PYTHONPATH=src python -m robosahaya.app --port 7865
   ```

2. **"rclpy ModuleNotFoundError in virtual environment"**  
   Make sure the virtual environment was created with `--system-site-packages` so that `/opt/ros/jazzy/lib/python3.12/site-packages` is visible.

3. **"How do I test Telugu-English code-mixed queries?"**  
   In the UI text box or voice recorder, try:
   - *"నా robot tirugutondi kani laser scan data ravatledu, why?"*
   - *"slash can lo publishers enduku levu?"*
   - *"camera link transform base link tho connect avvatledu"*
