import json
import random
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

# Import local modules from driver-safety project
from .drowsiness import DrowsinessDetector
from .rl_agent import AlertAgent, ACTIONS

# ==============================================================
# OpenEnv Pydantic Spec
# ==============================================================

class State(BaseModel):
    ear: float = Field(..., description="Eye Aspect Ratio: indicates eye openness.")
    drowsiness_state: str = Field(..., description="Current state (ALERT, DROWSY, MICROSLEEP).")
    ignore_count: int = Field(..., description="Number of times driver ignored past alerts.")
    sway_variance: float = Field(..., description="Simulated head sway variance for drunk detection.")
    eye_asymmetry: float = Field(..., description="Simulated eye asymmetry for drunk detection.")
    drunk_status: str = Field(..., description="Simulated drunk status (SOBER, DRUNK).")

class Action(BaseModel):
    action_type: str = Field(..., description="Type of alert action to take (BEEP, VOICE, ALARM, NONE)")

class StepResult(BaseModel):
    state: State
    reward: float
    done: bool
    info: Dict[str, Any]

# ==============================================================
# OpenEnv Environment
# ==============================================================

class DriverSafetyEnv:
    def __init__(self, task_level: str = "hard"):
        """
        Creates the openenv runtime environment.
        :param task_level: 'easy', 'medium', or 'hard'
        """
        self.task_level = task_level.lower()
        self.detector = DrowsinessDetector()
        self.agent = AlertAgent()
        
        self.step_count = 0
        self.max_steps = 100
        self.simulated_time = 0.0
        self.current_state = self._simulate_initial_state()

    def _simulate_initial_state(self) -> State:
        return State(
            ear=0.35, # Healthy open eyes
            drowsiness_state="ALERT",
            ignore_count=0,
            sway_variance=0.01,
            eye_asymmetry=0.0,
            drunk_status="SOBER"
        )
            
    def reset(self) -> State:
        """
        OpenEnv generic reset interface.
        """
        self.detector = DrowsinessDetector()
        self.agent = AlertAgent()
        self.step_count = 0
        self.simulated_time = 0.0
        self.current_state = self._simulate_initial_state()
        return self.current_state

    def state(self) -> State:
        """
        Returns the Pydantic State model.
        """
        return self.current_state

    def _generate_synthetic_ear(self) -> float:
        """Simulates changing eye aspect ratio incrementally without requiring video."""
        if self.step_count < 20:
            return random.uniform(0.30, 0.40)      # Alert
        elif self.step_count < 60:
            return random.uniform(0.20, 0.26)      # Getting drowsy
        else:
            return random.uniform(0.10, 0.17)      # Falling asleep

    def _simulate_drunk_features(self) -> Tuple[float, float, str]:
        """Calculates sway variance and eye asymmetry to simulate Drunk state."""
        sway = 0.0
        asym = 0.0
        status = "SOBER"
        
        if self.task_level == "hard" and random.random() < 0.3:
            # Simulate impaired driver metrics
            sway = random.uniform(0.06, 0.15)
            asym = random.uniform(0.03, 0.08)
            status = "DRUNK" if (sway > 0.08 or asym > 0.05) else "SOBER"
        else:
            sway = random.uniform(0.00, 0.04)
            asym = random.uniform(0.00, 0.02)
            
        return round(sway, 3), round(asym, 3), status

    def step(self, action: Action) -> StepResult:
        """
        The OpenEnv main interaction loop.
        """
        self.step_count += 1
        self.simulated_time += 0.5 # 0.5s per step
        
        # 1. Update RL agent logic based on the driver's response to the new action
        # If the driver was previously drowsy and an action is provided, we award it based on RL rules
        old_state_str = self.current_state.drowsiness_state
        action_idx = 0
        reward = 0.0
        
        if action.action_type in ACTIONS:
            action_idx = ACTIONS.index(action.action_type)
            
        if old_state_str in ('DROWSY', 'MICROSLEEP'):
            if action.action_type in ACTIONS:
                self.agent.alert_sent(old_state_str, action_idx)
                # Simulated response: strong alerts successfully wake driver
                if action.action_type in ('VOICE', 'ALARM'):
                    self.agent.driver_responded()
                    reward += 10.0
                else: 
                    self.agent.driver_ignored()
                    reward -= 5.0
                    
        # 2. Transition physics
        new_ear = round(self._generate_synthetic_ear(), 3)
        new_drowsy_state = self.detector.update(new_ear, new_ear, pitch=0.0, current_time=self.simulated_time)
        sway, asym, drunk_status = self._simulate_drunk_features()
        
        new_state = State(
            ear=new_ear,
            drowsiness_state=new_drowsy_state,
            ignore_count=self.agent.ignore_count,
            sway_variance=sway,
            eye_asymmetry=asym,
            drunk_status=drunk_status
        )
        self.current_state = new_state
        
        # Grade the state representing task score (0 to 1) 
        score = self.evaluate(new_state)
        
        return StepResult(
            state=new_state,
            reward=reward,
            done=(self.step_count >= self.max_steps),
            info={"score": score, "task": self.task_level}
        )
        
    def evaluate(self, current_state: State) -> float:
        """
        Grader for each task tracking safety index (0-1).
        """
        if self.task_level == "easy":
            # Easy task grader (Eye open/closed detection)
            return 1.0 if current_state.ear >= 0.25 else 0.0 
            
        elif self.task_level == "medium":
            # Medium task grader (drowsy vs alert score)
            if current_state.drowsiness_state == 'ALERT':
                return 1.0
            elif current_state.drowsiness_state == 'DROWSY':
                return 0.5
            return 0.0 # MICROSLEEP
            
        elif self.task_level == "hard":
            # Hard task grader (full safety: drowsy + drunk tracking)
            score = 1.0
            if current_state.drowsiness_state == 'DROWSY':
                score -= 0.3
            elif current_state.drowsiness_state == 'MICROSLEEP':
                score -= 0.5
                
            if current_state.drunk_status == 'DRUNK':
                score -= 0.5
                
            return max(0.0, round(score, 2))



