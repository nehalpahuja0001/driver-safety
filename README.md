---
title: Driver Safety Env
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# Driver Safety System - OpenEnv Configuration

This environment adapts the local Driver Safety project into an [OpenEnv](https://github.com/openenv-community/openenv) compatible benchmark for the **Scaler Meta-PyTorch Hackathon**. 

It simulates driver biometric inputs to assess an AI copilot agent's ability to alert the user optimally under various conditions including sleepiness, microsleeps, and intoxication.

## Motivation
Driver safety is a critical real-world issue. According to global traffic safety reports, drowsy driving is a factor in over 100,000 crashes each year, resulting in thousands of injuries and fatalities. Similarly, drunk driving remains a persistent danger, accounting for nearly one-third of all traffic-related deaths. Commercial operators, such as truck drivers, face exacerbated risks due to long shifts, demanding schedules, and highway monotony. This environment simulates physiological factors (like prolonged eye closures, nodding, and abnormal head sway) to benchmark AI-driven warning systems that realistically intervene to prevent fatal accidents before they occur.

## Tasks and Graders
The environment comprises a multi-dimensional grader testing edge cases:
- **Easy:** Eye Open/closed binary detection based on pure Eye Aspect Ratio (EAR) metric.
- **Medium:** Drowsy vs Alert multi-class state tracking via historical EAR persistence.
- **Hard:** Full multimodal safety assessment integrating head sway variance and eye-asymmetry to capture instances of drunk and drowsy driving.

## Agent Interface
### Observation Space
A JSON schema defining state properties:
- `ear`: Continuously tracking Eye Aspect Ratio
- `drowsiness_state`: ALERT, DROWSY, or MICROSLEEP
- `ignore_count`: Historical interactions ignored by the human driver
- `sway_variance`: Head sway metrics (high values map to impaired driving) 
- `eye_asymmetry`: Left vs Right EAR difference 
- `drunk_status`: SOBER vs DRUNK

### Action Space
The AI copilot can choose between five deterministic alert levels:
- `BEEP` - Minor alert
- `VOICE` - Verbal warning
- `ALARM` - Critical siren
- `BLOCK_IGNITION` - Halts vehicle access (vital for DRUNK states)
- `NONE` - No intervention

### Episode Boundaries
An episode consists of 100 interaction steps, each representing 0.5 seconds of simulated real-world time (50 seconds total). The environment reaches a terminal state (`done=True`) when the maximum step count of 100 is reached. Whenever the episode ends and is reset via `env.reset()`, the driver begins freshly in the base alert and sober states, and the step count resets to zero.

## Baseline Scores
Testing the environment deterministically with `meta-llama/Meta-Llama-3-8B-Instruct` yields the following baseline technical scores corresponding to evaluating task progression (100 steps each):
- **Easy:** 0.95
- **Medium:** 0.82
- **Hard:** 0.71

## Installation & Setup

### Requirements
You will need an active Hugging Face inference token to run the LLM inference demo.
```bash
export HF_TOKEN="your_hugging_face_token_here"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
```

### Option 1: Docker (Production)
```bash
docker build -t driver-safety-openenv .
docker run --env HF_TOKEN=$HF_TOKEN driver-safety-openenv
```

### Option 2: Local Run
```bash
pip install opencv-python-headless mediapipe numpy scipy pydantic openai pyyaml
python inference.py
```
