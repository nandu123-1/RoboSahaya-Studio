# 🛠️ RoboSahaya — Setup and Verification Guide

This guide provides end-to-end instructions for running and verifying **RoboSahaya** on your machine.

---

## 💻 System Requirements

| Mode | Supported Operating Systems | Requirements |
|---|---|---|
| **Mock Replay Mode (`--mode mock`)** | Ubuntu, Debian, macOS, Windows 10/11 | Python 3.11 or 3.12 (Zero ROS 2 dependencies needed) |
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
# Gnani AI Voice Services (https://app.gnani.ai/voice)
GNANI_API_KEY=your_gnani_api_key_here
GNANI_BASE_URL=https://api.gnani.ai/v1

# Execution Mode ('mock' for evaluator replay, 'live' for real robot)
ROBOSAHAYA_MODE=mock
ROBOSAHAYA_MOCK_SCENARIO=qos_reliability_mismatch

# ROS Domain
ROS_DOMAIN_ID=0
```

> 💡 **Offline / Evaluation Note:** If `GNANI_API_KEY` is not provided, RoboSahaya automatically runs with offline speech simulation and full text-based question input so judges can evaluate the entire diagnostic pipeline without needing an active API account.

---

## 🚀 3. Launching the Application

### Option A: Mock Replay Mode (Recommended for Judges)
Runs with zero external robotics hardware or ROS 2 dependencies. Replays exact ROS 2 graph snapshots:
```bash
python src/robosahaya/app.py --mode mock
```
Open **`http://127.0.0.1:7860`** in your browser.

Use the dropdown in the UI to switch between the 5 scenarios:
1. `qos_reliability_mismatch`: LiDAR publisher `BEST_EFFORT` vs Subscriber `RELIABLE`
2. `missing_laser_pub`: Topic `/scan` exists, but 0 publishers active
3. `tf_broken_camera_link`: Disconnected `base_link` $\rightarrow$ `camera_link` coordinate frame
4. `sim_time_divergence`: Paused `/clock` vs `use_sim_time: false` on node
5. `clean_healthy_system`: Fully operational robot

### Option B: Live ROS 2 Mode (Real Hardware or Gazebo)
To inspect an active physical robot or Gazebo simulation running on your machine:
```bash
# Source ROS 2 underlay
source /opt/ros/jazzy/setup.bash

# Ensure same domain ID as your robot
export ROS_DOMAIN_ID=0

# Launch RoboSahaya in live mode
python src/robosahaya/app.py --mode live
```

---

## 🧪 4. Automated Verification & Benchmarking

### Run the Pytest Suite
```bash
pytest tests/ -v
```
All 17 unit and integration tests run in under 0.2 seconds.

### Run the Performance Benchmark
```bash
PYTHONPATH=src python evaluation/run_benchmark.py
```
This tests all 5 labelled cases against ground truth, verifying:
- **Top-1 Diagnostic Accuracy** ($\ge 85\%$)
- **Evidence Traceability** ($\ge 95\%$)
- **Command Catalogue Safety Compliance** ($100\%$)
- **Telugu-English Code-Mixed Grounding Accuracy** ($\ge 80\%$)
- **P50 and P95 Execution Latency** ($<1\text{ ms}$)

---

## 🔍 5. Troubleshooting & FAQ

1. **"Gradio web page is blank or not opening"**  
   Ensure port 7860 is not in use by another service. You can specify a custom port:
   ```bash
   python src/robosahaya/app.py --port 7865
   ```

2. **"rclpy ModuleNotFoundError in virtual environment"**  
   Make sure the virtual environment was created with `--system-site-packages` so that `/opt/ros/jazzy/lib/python3.12/site-packages` is visible.

3. **"How do I test Telugu-English code-mixed queries?"**  
   In the UI text box or voice recorder, try:
   - *"నా robot tirugutondi kani laser scan data ravatledu, why?"*
   - *"slash can lo publishers enduku levu?"*
   - *"camera link transform base link tho connect avvatledu"*
