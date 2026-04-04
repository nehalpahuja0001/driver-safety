import time
from collections import deque
import numpy as np

EAR_DROWSY = 0.25

class DrowsinessDetector:
    def __init__(self):
        self.eyes_closed_start = None
        self.state    = 'ALERT'

        # 30-second PERCLOS
        self.history_30s = deque()
        
        # Sway tracking
        self.yaw_hist = deque()

        # Public metrics
        self.perclos = 0.0

    def update(self, left_ear, right_ear, pitch=0.0, yaw=0.0, current_time=None):
        if left_ear is None or right_ear is None:
            return self.state
        if current_time is None:
            current_time = time.time()

        ear = (left_ear + right_ear) / 2.0

        # Maintain 30-second history for PERCLOS
        self.history_30s.append((current_time, ear))
        while self.history_30s and current_time - self.history_30s[0][0] > 30.0:
            self.history_30s.popleft()

        # Maintain yaw history for 10 seconds to calculate sway
        self.yaw_hist.append((current_time, yaw))
        while self.yaw_hist and current_time - self.yaw_hist[0][0] > 10.0:
            self.yaw_hist.popleft()

        # Calculate PERCLOS over 30s
        perclos_frames = sum(1 for _, e in self.history_30s if e < EAR_DROWSY)
        self.perclos = (perclos_frames / len(self.history_30s)) if len(self.history_30s) > 0 else 0.0

        # Head sway (variance of yaw over last 10s)
        sway_critical = False
        if len(self.yaw_hist) > 15:
            yaws = [y for _, y in self.yaw_hist]
            if np.std(yaws) > 5.0:
                sway_critical = True

        # Rule 1: CRITICAL (Passed out)
        if self.perclos > 0.80 or sway_critical:
            self.state = 'CRITICAL'
            self.eyes_closed_start = None
            return self.state

        # Rule 2: NODDING OFF (Head tilted forward)
        if pitch > 20.0:
            self.state = 'NODDING OFF'
            self.eyes_closed_start = None
            return self.state

        if ear < EAR_DROWSY:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            elapsed = current_time - self.eyes_closed_start
            
            if elapsed >= 1.0 and self.perclos > 0.70:
                self.state = 'MICROSLEEP'
            elif elapsed >= 2.0:
                self.state = 'DROWSY'
            else:
                self.state = 'ALERT'
        else:
            self.eyes_closed_start = None
            self.state = 'ALERT'

        return self.state

    def get_perclos(self):
        return round(self.perclos * 100, 1)

