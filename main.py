import cv2
import time
import numpy as np
import threading
import winsound
import pyttsx3
import scipy.io.wavfile as wavfile
import os

from eye_detector import get_face_features
from drowsiness import DrowsinessDetector
from rl_agent import AlertAgent, ACTIONS

# GENERATE ALARM WAV
def generate_alarm_file():
    if not os.path.exists("alarm.wav"):
        sample_rate = 44100
        duration = 1.0 # 1 second loop
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = np.sin(2 * np.pi * 2500 * t)
        wave_16bit = np.int16(wave * 32767)
        wavfile.write("alarm.wav", sample_rate, wave_16bit)

generate_alarm_file()

detector = DrowsinessDetector()
agent    = AlertAgent()
cap      = cv2.VideoCapture(0)

alert_time  = None
drowsy_start_time = None
ALERT_WAIT  = 5.0
flash_action = -1
alarm_active = False
tts_lock = threading.Lock()

def play_voice_alert():
    if not tts_lock.acquire(blocking=False):
        return
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = pyttsx3.init()
        engine.say("Warning! Driver drowsy! Please stop the vehicle!")
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        tts_lock.release()

def play_alert(action):
    global alarm_active
    if action == 0:
        # BEEP: Only 2 short beeps
        winsound.Beep(800, 200)
        time.sleep(0.1)
        winsound.Beep(800, 200)
    elif action == 1:
        # VOICE: pyttsx3 saying warning
        play_voice_alert()
    elif action == 2:
        # ALARM: Loud continuous alarm using winsound
        if not alarm_active:
            alarm_active = True
            winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)

def stop_continuous_alarm():
    global alarm_active
    if alarm_active:
        winsound.PlaySound(None, winsound.SND_PURGE)
        alarm_active = False

def flash_screen(action):
    global flash_action
    flashes = 2 if action == 0 else 5 if action == 1 else 15
    for _ in range(flashes):
        flash_action = action
        time.sleep(0.15)
        flash_action = -1
        time.sleep(0.1)

def trigger_alert(state):
    global alert_time
    action     = agent.choose_action(state)
    alert_type = ACTIONS[action]
    agent.alert_sent(state, action)
    alert_time = time.time()
    
    print(f"IGNORE COUNT: {agent.ignore_count}, PLAYING: {alert_type}")
    
    threading.Thread(
        target=play_alert, args=(action,), daemon=True
    ).start()
    threading.Thread(
        target=flash_screen, args=(action,), daemon=True
    ).start()

print("[SYSTEM] Driver Safety System Starting...")
print("[SYSTEM] Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, pitch, frame = get_face_features(frame)
    state = detector.update(ear, pitch) if ear is not None else 'NO_FACE'

    # Red flash overlay
    if flash_action >= 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0),
                      (frame.shape[1], frame.shape[0]),
                      (0, 0, 255), -1)
        alpha = 0.9 if flash_action == 2 else 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # EAR progress bar
    if ear is not None:
        bar_w = int(ear * 400)
        bar_color = (0,255,0) if ear > 0.25 else \
                    (0,165,255) if ear > 0.18 else (0,0,255)
        cv2.rectangle(frame, (20, 200), (420, 225), (50,50,50), -1)
        cv2.rectangle(frame, (20, 200), (20+bar_w, 225), bar_color, -1)
        cv2.putText(frame, "EAR", (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    ear_text = f"EAR: {ear}" if ear is not None else "EAR: --"
    pitch_text = f"PITCH: {pitch:.1f}" if pitch is not None else "PITCH: --"
    
    color = (0,255,0) if state == 'ALERT' else \
            (0,165,255) if state == 'DROWSY' else (0,0,255)

    cv2.putText(frame, ear_text,
        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, pitch_text,
        (220,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"STATE: {state}",
        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Ignores: {agent.ignore_count}",
        (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # State tracking logic for RL ignore escalation
    if state in ('DROWSY', 'MICROSLEEP'):
        if drowsy_start_time is None:
            # Entered bad state
            drowsy_start_time = time.time()
            trigger_alert(state)
        else:
            # Check if it has been 5 continuous seconds
            if (time.time() - drowsy_start_time) >= ALERT_WAIT:
                agent.driver_ignored()
                # Reset counter and trigger a higher level alert
                drowsy_start_time = time.time() 
                trigger_alert(state)
                
        cv2.putText(frame, f"ALERT: {ACTIONS[agent.last_action or 0]}",
            (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    elif state == 'ALERT':
        if drowsy_start_time is not None:
            # Driver recovered
            agent.driver_responded()
            drowsy_start_time = None
            alert_time = None
            stop_continuous_alarm()

    cv2.imshow("Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
