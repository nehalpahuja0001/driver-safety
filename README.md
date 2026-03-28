# Driver Safety System - OpenEnv Configuration

This environment adapts the local Driver Safety project into an [OpenEnv](https://github.com/openenv-community/openenv) compatible benchmark for the **Scaler Meta-PyTorch Hackathon**. 

It simulates driver biometric inputs to assess an AI copilot agent's ability to alert the user optimally under various conditions including sleepiness, microsleeps, and intoxication.

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
The AI copilot can choose between four deterministic alert levels:
- `BEEP` - Minor alert
- `VOICE` - Verbal warning
- `ALARM` - Critical siren
- `NONE` - No intervention

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
