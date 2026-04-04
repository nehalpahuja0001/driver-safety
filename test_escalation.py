import time
from server.drowsiness import DrowsinessDetector
from server.rl_agent import AlertAgent, ACTIONS

def test_escalation():
    print("--- Testing Drowsiness Escalation ---")
    detector = DrowsinessDetector()
    agent = AlertAgent()
    
    agent.ignore_count = 0
    agent.last_state = None
    
    print("1. Feeding NORMAL frames")
    for _ in range(30):
        detector.update(0.35, 0.0)
    print("Queue filled with normal frames.")
    
    print("\n2. Driver enters DROWSY state. Simulating frames without recovery...")
    drowsy_start_time = None
    simulated_time = 0.0
    time_step = 0.5 # Wait, if time_step is 0.5s per frame, it only does 2 frames per second. But the detector expects 20 frames to trigger. In real time that's 20 frames at 30fps < 1 sec. 
    # Let's realistically simulate 10 fps.
    time_step = 0.1
    
    for step in range(150): # 15 seconds worth of frames at 10 fps
        state = detector.update(0.15, 0.0) # very low ear
        
        if state == 'DROWSY' or state == 'MICROSLEEP':
            if drowsy_start_time is None:
                drowsy_start_time = simulated_time
                agent.alert_sent(state, agent.choose_action(state))
                print(f"[t={simulated_time:.1f}s] Entered {state}. Initial Alert: {ACTIONS[agent.last_action]}")
            else:
                if (simulated_time - drowsy_start_time) >= 5.0:
                    agent.driver_ignored()
                    drowsy_start_time = simulated_time
                    action = agent.choose_action(state)
                    agent.alert_sent(state, action)
                    print(f"[t={simulated_time:.1f}s] Alert ignored for 5s! Escaping... Logged Ignore Count: {agent.ignore_count}")
                    print(f"              New Alert Level: {ACTIONS[action]}")
                    
        simulated_time += time_step
        
    print("\n3. Driver recovers")
    for _ in range(50):
        state = detector.update(0.35, 0.0)
        if state == 'ALERT' and drowsy_start_time is not None:
            agent.driver_responded()
            drowsy_start_time = None
            print(f"[t={simulated_time:.1f}s] Recovered! Ignore count reduced to: {agent.ignore_count}. Current State: {state}")
            break
        simulated_time += time_step
            
    print("\n--- Test finished ---")

if __name__ == "__main__":
    test_escalation()

