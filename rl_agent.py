import numpy as np
import json
import os

ACTIONS  = ['BEEP', 'VOICE', 'ALARM']
ALPHA    = 0.1
GAMMA    = 0.9
EPSILON  = 0.0
Q_FILE   = 'q_table.json'

class AlertAgent:
    def __init__(self):
        self.q             = {}
        self.ignore_count  = 0
        self.last_state    = None
        self.last_action   = None
        self.load()

    def _key(self, state):
        ic = min(self.ignore_count, 5)
        return f'{state}_{ic}'

    def get_q(self, state):
        k = self._key(state)
        if k not in self.q:
            self.q[k] = [0.0, 0.0, 0.0]
        return self.q[k]

    def choose_action(self, state):
        if np.random.random() < EPSILON:
            return np.random.randint(3)
        ic = min(self.ignore_count, 2)
        return ic

    def update(self, state, action, reward, next_state):
        q_vals = self.get_q(state)
        q_next = max(self.get_q(next_state))
        q_vals[action] += ALPHA * (reward + GAMMA * q_next - q_vals[action])
        self.q[self._key(state)] = q_vals
        self.save()

    def alert_sent(self, state, action):
        self.last_state  = state
        self.last_action = action

    def driver_responded(self):
        if self.last_state:
            self.update(self.last_state, self.last_action, +10, 'ALERT')
            self.ignore_count = max(0, self.ignore_count - 1)
            self.last_state = None

    def driver_ignored(self):
        if self.last_state:
            self.update(self.last_state, self.last_action, -5, self.last_state)
            self.ignore_count += 1
            self.last_state = None

    def save(self):
        with open(Q_FILE, 'w') as f:
            json.dump(self.q, f)

    def load(self):
        if os.path.exists(Q_FILE):
            with open(Q_FILE) as f:
                self.q = json.load(f)
