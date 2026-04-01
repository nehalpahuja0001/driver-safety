import time
from collections import deque

EAR_DROWSY = 0.25
EAR_SLEEP  = 0.15

class DrowsinessDetector:
    def __init__(self):
        self.eyes_closed_start = None
        self.ear_hist = deque(maxlen=30)
        self.state    = 'ALERT'

    def update(self, ear, pitch=0.0, current_time=None):
        if ear is None:
            return self.state
        if current_time is None:
            current_time = time.time()

        self.ear_hist.append(ear)

        if ear < EAR_DROWSY:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            elapsed = current_time - self.eyes_closed_start
            if elapsed >= 2.5:
                if ear < EAR_SLEEP:
                    self.state = 'MICROSLEEP'
                else:
                    self.state = 'DROWSY'
        else:
            # Aankhein khul gayi — TURANT reset
            self.eyes_closed_start = None
            self.state = 'ALERT'

        return self.state

    def get_perclos(self):
        if not self.ear_hist:
            return 0
        closed = sum(1 for e in self.ear_hist if e < 0.20)
        return round(closed / len(self.ear_hist) * 100, 1)
