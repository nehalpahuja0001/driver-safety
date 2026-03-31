from collections import deque

EAR_DROWSY    = 0.25
EAR_SLEEP     = 0.18
PITCH_NOD_THRESHOLD = 15.0
CONSEC_FRAMES = 20

class DrowsinessDetector:
    def __init__(self):
        self.counter   = 0
        self.ear_hist  = deque(maxlen=30)
        self.state     = 'ALERT'

    def update(self, ear, pitch):
        if ear is None or pitch is None:
            return self.state

        self.ear_hist.append(ear)
        
        # Calculate moving average
        avg_ear = sum(self.ear_hist) / len(self.ear_hist)
        
        # Avoid false positives initially
        if len(self.ear_hist) < 10:
            avg_ear = ear
            
        is_nodding = abs(pitch) > PITCH_NOD_THRESHOLD

        if avg_ear < EAR_SLEEP or is_nodding:
            self.counter += 1
            if self.counter >= CONSEC_FRAMES:
                self.state = 'MICROSLEEP'
        elif avg_ear < EAR_DROWSY:
            self.counter += 1
            if self.counter >= CONSEC_FRAMES // 2:
                self.state = 'DROWSY'
        else:
            self.counter = max(0, self.counter - 2)
            if self.counter == 0:
                self.state = 'ALERT'
        return self.state

    def get_perclos(self):
        if not self.ear_hist:
            return 0
        closed = sum(1 for e in self.ear_hist if e < 0.20)
        return round(closed / len(self.ear_hist) * 100, 1)