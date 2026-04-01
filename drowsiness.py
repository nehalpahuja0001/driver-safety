import time
from collections import deque

EAR_DROWSY    = 0.25
PITCH_NOD_THRESHOLD = 15.0

class DrowsinessDetector:
    def __init__(self):
        self.eyes_closed_start = None
        self.ear_hist  = deque(maxlen=30)
        self.state     = 'ALERT'

    def update(self, ear, pitch=0.0, current_time=None):
        if ear is None or pitch is None:
            return self.state

        if current_time is None:
            current_time = time.time()

        self.ear_hist.append(ear)
        is_nodding = abs(pitch) > PITCH_NOD_THRESHOLD

        if ear < EAR_DROWSY or is_nodding:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            else:
                if (current_time - self.eyes_closed_start) >= 2.5:
                    self.state = 'DROWSY'
        else:
            self.eyes_closed_start = None
            self.state = 'ALERT'
            
        return self.state

    def get_perclos(self):
        if not self.ear_hist:
            return 0
        closed = sum(1 for e in self.ear_hist if e < 0.20)
        return round(closed / len(self.ear_hist) * 100, 1)