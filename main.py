import cv2
import time
import numpy as np
import threading
import winsound
import pyttsx3
import scipy.io.wavfile as wavfile
import os
from datetime import datetime

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

system_start_time   = time.time()
drowsy_start_time   = None
recovery_start_time = None
flash_action        = -1
alarm_active        = False
alert_active        = False
tts_lock            = threading.Lock()
voice_engine        = None

def stop_all_sounds():
    global alarm_active, alert_active, flash_action, voice_engine
    for _ in range(3):
        winsound.PlaySound(None, winsound.SND_PURGE)
    if voice_engine is not None:
        try:
            voice_engine.stop()
        except:
            pass
    alarm_active  = False
    alert_active  = False
    flash_action  = -1

def play_voice_alert():
    global voice_engine
    if not tts_lock.acquire(blocking=False):
        return
    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        voice_engine = pyttsx3.init()
        voice_engine.say("Warning! Driver drowsy! Stay alert!")
        voice_engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        voice_engine = None
        tts_lock.release()

def play_alert(action):
    global alarm_active, alert_active
    if not alert_active:
        return
    if action == 0:
        play_voice_alert()
    elif action == 1:
        if not alarm_active:
            alarm_active = True
            winsound.PlaySound("alarm.wav",
                winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)

def flash_screen(action):
    global flash_action
    flashes = 5 if action == 0 else 15
    for _ in range(flashes):
        if not alert_active:
            flash_action = -1
            return
        flash_action = action
        time.sleep(0.15)
        flash_action = -1
        time.sleep(0.1)

def trigger_alert(agent_state):
    global alert_active
    alert_active = True
    
    # RL agent will dynamically decide the best action (BEEP/VOICE/ALARM)
    # based on the state (Drowsy vs Microsleep vs Drunk), context, and user history!
    action = agent.choose_action(agent_state)
        
    agent.alert_sent(agent_state, action)
    print(f"IGNORE COUNT: {agent.ignore_count}, PLAYING: {ACTIONS[action]}")
    threading.Thread(target=play_alert, args=(action,), daemon=True).start()
    threading.Thread(target=flash_screen, args=(action,), daemon=True).start()

print("[SYSTEM] Driver Safety System Starting...")
print("[SYSTEM] Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left_ear, right_ear, pitch, yaw, frame = get_face_features(frame)
    if left_ear is not None:
        ear = (left_ear + right_ear) / 2.0
    else:
        ear = None

    state = detector.update(left_ear, right_ear, pitch, yaw) if left_ear is not None else 'NO_FACE'

    time_of_day = "Morning" if 6 <= datetime.now().hour < 12 else "Afternoon" if 12 <= datetime.now().hour < 18 else "Night"
    agent_state = (state, agent.ignore_count, time_of_day)

    # --- DASHBOARD UI REDESIGN ---
    elapsed_session = int(time.time() - system_start_time)
    hrs, remainder = divmod(elapsed_session, 3600)
    mins, secs = divmod(remainder, 60)
    session_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

    canvas = np.full((800, 1200, 3), (10, 10, 10), dtype=np.uint8)
    
    theme_color = (0, 220, 0) if state == 'ALERT' else \
                  (0, 165, 255) if state in ('DROWSY', 'NODDING OFF') else (0, 0, 255)
    panel_bg = (30, 30, 30)
    text_color = (230, 230, 230)

    # TOP HEADER PANEL
    cv2.rectangle(canvas, (20, 20), (1180, 80), panel_bg, -1)
    cv2.putText(canvas, "DRIVER MONITORING SYSTEM", (40, 60), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, theme_color, 2)
    cur_time = datetime.now().strftime("%I:%M:%S %p")
    cv2.putText(canvas, f"TIME: {cur_time}   SESSION: {session_str}", 
                (700, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # CAMERA FEED PANEL
    cv2.rectangle(canvas, (20, 100), (660, 580), panel_bg, -1)
    if frame is not None:
        if flash_action >= 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), -1)
            alpha = 0.9 if flash_action == 2 else 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        h, w = frame.shape[:2]
        if h > 480 or w > 640:
            frame = cv2.resize(frame, (640, 480))
            h, w = 480, 640
        canvas[100:100+h, 20:20+w] = frame

    # STATE PANEL
    cv2.rectangle(canvas, (680, 100), (1180, 250), panel_bg, -1)
    cv2.putText(canvas, "CURRENT STATE", (700, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    st_sz = cv2.getTextSize(state, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)[0]
    cv2.putText(canvas, state, (680 + (500 - st_sz[0]) // 2, 210), cv2.FONT_HERSHEY_DUPLEX, 2.0, theme_color, 3)

    # METRICS PANEL
    cv2.rectangle(canvas, (680, 270), (1180, 580), panel_bg, -1)
    cv2.putText(canvas, "BIOMETRICS", (700, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    ear_val = ear if ear is not None else 0.0
    cv2.putText(canvas, f"EAR (Eye Aspect Ratio): {ear_val:.3f}", (700, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    bar_w = min(int(ear_val * 1000), 460)
    cv2.rectangle(canvas, (700, 375), (1160, 400), (50, 50, 50), -1)
    cv2.rectangle(canvas, (700, 375), (700+max(bar_w, 0), 400), theme_color, -1)
    
    perclos = detector.get_perclos()
    cv2.putText(canvas, f"PERCLOS: {perclos}%", (700, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    p_color = (0, 0, 255) if perclos > 50 else (0, 165, 255) if perclos > 20 else (0, 255, 0)
    p_bar_w = min(int((perclos / 100.0) * 460), 460)
    cv2.rectangle(canvas, (700, 475), (1160, 500), (50, 50, 50), -1)
    cv2.rectangle(canvas, (700, 475), (700+max(p_bar_w, 0), 500), p_color, -1)
    
    p_txt = f"PITCH: {pitch:.1f}" if left_ear is not None else "PITCH: --"
    y_txt = f"YAW: {yaw:.1f}" if left_ear is not None else "YAW: --"
    cv2.putText(canvas, f"{p_txt}   {y_txt}", (700, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # ALERTS & IGNORE PANEL
    cv2.rectangle(canvas, (20, 600), (660, 780), panel_bg, -1)
    cv2.putText(canvas, "AGENT ALERTS", (40, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    b_color, b_text = ((0, 255, 0), "SAFE")
    if alert_active:
        if agent.last_action == 0: b_color, b_text = ((0, 165, 255), "WARNING (VOICE)")
        elif agent.last_action == 1: b_color, b_text = ((0, 0, 255), "ALARM")
    cv2.rectangle(canvas, (40, 660), (340, 740), b_color, -1)
    b_sz = cv2.getTextSize(b_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(canvas, b_text, (40 + (300 - b_sz[0])//2, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    cv2.rectangle(canvas, (680, 600), (1180, 780), panel_bg, -1)
    cv2.putText(canvas, "IGNORE COUNT", (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    for i in range(5):
        cx = 740 + i * 80
        cv2.circle(canvas, (cx, 710), 20, (100, 100, 100), 2)
        if i < agent.ignore_count:
            cv2.circle(canvas, (cx, 710), 16, (0, 0, 255), -1)

    if state in ('DROWSY', 'MICROSLEEP', 'NODDING OFF', 'CRITICAL'):
        recovery_start_time = None
        if drowsy_start_time is None:
            drowsy_start_time = time.time()
            trigger_alert(agent_state)
        else:
            if time.time() - drowsy_start_time >= 4.0 and agent.ignore_count == 0:
                agent.driver_ignored()
                agent_state = (state, agent.ignore_count, time_of_day)
                stop_all_sounds()
                trigger_alert(agent_state)
            else:
                # Re-trigger if state escalated
                last_dl = agent.last_state[0] if agent.last_state else None
                if state != last_dl:
                    stop_all_sounds()
                    trigger_alert(agent_state)

    elif state == 'ALERT':
        if drowsy_start_time is not None:
            stop_all_sounds()
            agent.driver_responded()
            drowsy_start_time   = None
            recovery_start_time = None

    cv2.imshow("DRIVER MONITORING SYSTEM - DASHBOARD", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
