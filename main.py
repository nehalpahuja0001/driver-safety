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

def generate_alarm_file():
    if not os.path.exists("alarm.wav"):
        sample_rate = 44100
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        wave = np.int16(np.sin(2 * np.pi * 2500 * t) * 32767)
        wavfile.write("alarm.wav", sample_rate, wave)

generate_alarm_file()

detector = DrowsinessDetector()
agent    = AlertAgent()
cap      = cv2.VideoCapture(0)

drowsy_start_time   = None
recovery_start_time = None
ALERT_WAIT          = 5.0
flash_action        = -1
alarm_active        = False
alert_active        = False
tts_lock            = threading.Lock()

def stop_all_sounds():
    global alarm_active, alert_active, flash_action
    for _ in range(3):
        winsound.PlaySound(None, winsound.SND_PURGE)
    alarm_active  = False
    alert_active  = False
    flash_action  = -1

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
    global alarm_active, alert_active
    if not alert_active:
        return
    if action == 0:
        winsound.Beep(800, 200)
        if alert_active:
            time.sleep(0.1)
            winsound.Beep(800, 200)
    elif action == 1:
        play_voice_alert()
    elif action == 2:
        if not alarm_active:
            alarm_active = True
            winsound.PlaySound("alarm.wav",
                winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)

def flash_screen(action):
    global flash_action
    flashes = 2 if action == 0 else 5 if action == 1 else 15
    for _ in range(flashes):
        if not alert_active:
            flash_action = -1
            return
        flash_action = action
        time.sleep(0.15)
        flash_action = -1
        time.sleep(0.1)

def trigger_alert(state):
    global alert_active
    alert_active = True
    action = agent.choose_action(state)
    agent.alert_sent(state, action)
    print(f"IGNORE COUNT: {agent.ignore_count}, PLAYING: {ACTIONS[action]}")
    threading.Thread(target=play_alert, args=(action,), daemon=True).start()
    threading.Thread(target=flash_screen, args=(action,), daemon=True).start()

print("[SYSTEM] Driver Safety System Starting...")
print("[SYSTEM] Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, pitch, frame = get_face_features(frame)
    state = detector.update(ear, pitch) if ear is not None else 'NO_FACE'

    # Flash overlay
    if flash_action >= 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0),
            (frame.shape[1], frame.shape[0]), (0,0,255), -1)
        alpha = 0.9 if flash_action == 2 else 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # EAR bar
    if ear is not None:
        bar_w = min(int(ear * 400), 400)
        bar_color = (0,255,0) if ear > 0.25 else \
                    (0,165,255) if ear > 0.18 else (0,0,255)
        cv2.rectangle(frame, (20,200), (420,225), (50,50,50), -1)
        cv2.rectangle(frame, (20,200), (20+bar_w,225), bar_color, -1)
        cv2.putText(frame, "EAR", (20,245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    ear_text   = f"EAR: {ear:.3f}" if ear is not None else "EAR: --"
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

    if state in ('DROWSY', 'MICROSLEEP'):
        recovery_start_time = None
        if drowsy_start_time is None:
            drowsy_start_time = time.time()
            trigger_alert(state)
        else:
            if (time.time() - drowsy_start_time) >= ALERT_WAIT:
                agent.driver_ignored()
                drowsy_start_time = time.time()
                stop_all_sounds()
                trigger_alert(state)

        if agent.last_action is not None:
            cv2.putText(frame, f"ALERT: {ACTIONS[agent.last_action]}",
                (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    elif state == 'ALERT':
        if drowsy_start_time is not None:
            # 3 baar purge karo — turant band karo
            for _ in range(3):
                winsound.PlaySound(None, winsound.SND_PURGE)
            alarm_active = False
            alert_active = False
            flash_action = -1

            if recovery_start_time is None:
                recovery_start_time = time.time()
            elif (time.time() - recovery_start_time) >= 1.0:
                agent.driver_responded()
                drowsy_start_time   = None
                recovery_start_time = None

    cv2.imshow("Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
