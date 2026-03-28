import cv2
import time
import numpy as np
from eye_detector import get_ear_from_frame
from drowsiness import DrowsinessDetector
from rl_agent import AlertAgent, ACTIONS
import threading
import winsound

detector = DrowsinessDetector()
agent    = AlertAgent()
cap      = cv2.VideoCapture(0)

alert_time  = None
ALERT_WAIT  = 3
flash_state = False

def play_alert(action):
    if action == 0:
        winsound.Beep(800, 400)
    elif action == 1:
        for _ in range(3):
            winsound.Beep(1200, 300)
            time.sleep(0.1)
    elif action == 2:
        for _ in range(8):
            winsound.Beep(2000, 150)
            time.sleep(0.05)

def flash_screen(action):
    global flash_state
    flashes = 2 if action == 0 else 4 if action == 1 else 8
    color   = (0,0,180) if action == 0 else (0,0,220) if action == 1 else (0,0,255)
    for _ in range(flashes):
        flash_state = True
        time.sleep(0.15)
        flash_state = False
        time.sleep(0.1)

print("[SYSTEM] Driver Safety System Starting...")
print("[SYSTEM] Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear, frame = get_ear_from_frame(frame)
    state = detector.update(ear) if ear else 'NO_FACE'

    # Red flash overlay
    if flash_state:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0),
                      (frame.shape[1], frame.shape[0]),
                      (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # EAR progress bar
    if ear:
        bar_w = int(ear * 400)
        bar_color = (0,255,0) if ear > 0.25 else \
                    (0,165,255) if ear > 0.18 else (0,0,255)
        cv2.rectangle(frame, (20, 200), (420, 225), (50,50,50), -1)
        cv2.rectangle(frame, (20, 200), (20+bar_w, 225), bar_color, -1)
        cv2.putText(frame, "EAR", (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    ear_text = f"EAR: {ear}" if ear else "EAR: --"
    color = (0,255,0) if state == 'ALERT' else \
            (0,165,255) if state == 'DROWSY' else (0,0,255)

    cv2.putText(frame, ear_text,
        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"STATE: {state}",
        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Ignores: {agent.ignore_count}",
        (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    alert_type = None
    if state in ('DROWSY', 'MICROSLEEP'):
        if alert_time is None:
            action     = agent.choose_action(state)
            alert_type = ACTIONS[action]
            agent.alert_sent(state, action)
            alert_time = time.time()
            threading.Thread(
                target=play_alert, args=(action,), daemon=True
            ).start()
            threading.Thread(
                target=flash_screen, args=(action,), daemon=True
            ).start()
        cv2.putText(frame, f"ALERT: {ACTIONS[agent.last_action or 0]}",
            (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    else:
        if alert_time and (time.time() - alert_time) > ALERT_WAIT:
            agent.driver_responded()
            alert_time = None

    if alert_time and state in ('DROWSY', 'MICROSLEEP'):
        if (time.time() - alert_time) > ALERT_WAIT:
            agent.driver_ignored()
            alert_time = None

    cv2.imshow("Driver Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
