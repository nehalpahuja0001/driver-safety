import numpy as np
import json
import os
import math

ACTIONS  = ['VOICE', 'ALARM']
ALPHA    = 0.1
GAMMA    = 0.9
EPSILON  = 0.2  # Real Q-learning utilizes e-greedy exploration
Q_FILE   = 'q_table.json'

class AlertAgent:
    def __init__(self):
        self.q             = {}
        self.ignore_count  = 0
        self.last_state    = None
        self.last_action   = None
        self.ignored_actions = {} # state -> {action: consecutive_ignores}
        self.load()

    def _key(self, state_tuple):
        # state is (drowsiness_level, ignore_count, time_of_day)
        if isinstance(state_tuple, tuple):
            dl, ic, tod = state_tuple
            dl = str(dl)
            ic = min(ic, 5) # Cap ignore count
            return f'{dl}_{ic}_{tod}'
        return str(state_tuple)

    def get_q(self, state_tuple):
        k = self._key(state_tuple)
        if k not in self.q:
            self.q[k] = [0.0, 0.0, 0.0]
        return self.q[k]

    def choose_action(self, state_tuple):
        # Strict safety logic: VOICE first, ALARM second
        ic = state_tuple[1] if isinstance(state_tuple, tuple) else self.ignore_count
        if ic == 0:
            return 0 # VOICE
        else:
            return 1 # ALARM

    def update(self, state_tuple, action, reward, next_state_tuple):
        q_vals = self.get_q(state_tuple)
        q_next = self.get_q(next_state_tuple) if next_state_tuple else [0.0, 0.0, 0.0]
        max_q_next = max(q_next)
        
        q_vals[action] += ALPHA * (reward + GAMMA * max_q_next - q_vals[action])
        self.q[self._key(state_tuple)] = q_vals
        self.save()

    def alert_sent(self, state_tuple, action):
        self.last_state  = state_tuple
        self.last_action = action

    def driver_responded(self):
        if self.last_state is not None and self.last_action is not None:
            # +10 if driver recovers
            self.update(self.last_state, self.last_action, +10, 'ALERT')
            self.ignore_count = max(0, self.ignore_count - 1)
            
            # Reset the ignored_actions counter for this specific state-action because it finally worked
            dl = self.last_state[0] if isinstance(self.last_state, tuple) else self.last_state
            if dl in self.ignored_actions:
                self.ignored_actions[dl][self.last_action] = 0
                
            self.last_state = None
            self.last_action = None

    def driver_ignored(self):
        if self.last_state is not None and self.last_action is not None:
            # -5 if ignored
            self.update(self.last_state, self.last_action, -5, self.last_state)
            self.ignore_count += 1
            
            # Increment adaptive ignore counter
            dl = self.last_state[0] if isinstance(self.last_state, tuple) else self.last_state
            if dl not in self.ignored_actions:
                self.ignored_actions[dl] = {0: 0, 1: 0, 2: 0}
            self.ignored_actions[dl][self.last_action] += 1
            
            self.last_state = None
            self.last_action = None

    def driver_unnecessary_alert(self):
        # -2 for unnecessary alerts (e.g. system alarmed but state changed back to alert almost immediately on its own)
        if self.last_state is not None and self.last_action is not None:
            self.update(self.last_state, self.last_action, -2, 'ALERT')
            self.last_state = None
            self.last_action = None

    def save(self):
        with open(Q_FILE, 'w') as f:
            json.dump(self.q, f)

    def load(self):
        if os.path.exists(Q_FILE):
            try:
                with open(Q_FILE) as f:
                    self.q = json.load(f)
            except:
                self.q = {}
