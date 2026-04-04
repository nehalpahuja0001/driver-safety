import os
from openai import OpenAI
from server.environment import DriverSafetyEnv, Action, State

def get_action_from_llm(client: OpenAI, model_name: str, state: State) -> str:
    # Hardcoded safety intercept for DRUNK status without needing LLM reasoning
    if state.drunk_status == "DRUNK":
        return "BLOCK_IGNITION"
        
    prompt = f"""
    You are an AI driver safety co-pilot.
    Current driver state:
    - Eye Aspect Ratio (EAR): {state.ear}
    - Drowsiness: {state.drowsiness_state}
    - Past ignored alerts: {state.ignore_count}
    - Sway Variance: {state.sway_variance}
    - Eye Asymmetry: {state.eye_asymmetry}
    - Drunk Status: {state.drunk_status}
    
    If Drowsiness is 'DROWSY', choose 'BEEP' or 'VOICE'.
    If Drowsiness is 'MICROSLEEP', choose 'ALARM'.
    If driver is 'ALERT', choose 'NONE'.
    
    Choose ONE action: BEEP, VOICE, ALARM, BLOCK_IGNITION, or NONE.
    Only reply with the exact action string.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful co-pilot AI. Reply only with action types: BEEP, VOICE, ALARM, BLOCK_IGNITION, NONE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42
        )
        action_raw = response.choices[0].message.content.strip().upper()
        # Clean potential conversational text
        for a in ["BLOCK_IGNITION", "ALARM", "VOICE", "BEEP", "NONE"]:
            if a in action_raw:
                return a
        return "NONE"
    except Exception as e:
        return "NONE"

def main():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    print("START")
    tasks = ['easy', 'medium', 'hard']
    for t in tasks:
        env = DriverSafetyEnv(task_level=t)
        state = env.reset()
        
        for step_num in range(100): # Demo 100 steps per task
            if HF_TOKEN:
                action_str = get_action_from_llm(client, model_name, state)
            else:
                action_str = "NONE"
                if t == 'easy':
                    if state.drowsiness_state == 'DROWSY': action_str = 'VOICE'
                elif t == 'medium':
                    if state.drowsiness_state in ('CRITICAL', 'MICROSLEEP'): action_str = 'ALARM'
                    elif state.drowsiness_state == 'DROWSY': action_str = 'VOICE'
                elif t == 'hard':
                    if state.drunk_status == 'DRUNK': action_str = 'BLOCK_IGNITION'
                    elif state.drowsiness_state in ('CRITICAL', 'MICROSLEEP'): action_str = 'ALARM'
                    elif state.drowsiness_state == 'DROWSY': action_str = 'VOICE'
                
            if action_str not in ["BEEP", "VOICE", "ALARM", "BLOCK_IGNITION", "NONE"]:
                action_str = "NONE"
                
            action_obj = Action(action_type=action_str)
            result = env.step(action_obj)
            
            print(f"STEP {step_num}")
            
            state = result.state
            if result.done:
                break
    print("END")

if __name__ == "__main__":
    main()
