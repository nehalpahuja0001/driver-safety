import os
from openai import OpenAI
from environment import DriverSafetyEnv, Action, State

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
            temperature=0.0
        )
        action_raw = response.choices[0].message.content.strip().upper()
        # Clean potential conversational text
        for a in ["BLOCK_IGNITION", "ALARM", "VOICE", "BEEP", "NONE"]:
            if a in action_raw:
                return a
        return "NONE"
    except Exception as e:
        print(f"[ERROR] LLM Request failed: {e}")
        return "NONE"

def main():
    api_base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Please set the HF_TOKEN environment variable (e.g. export HF_TOKEN='your_token').")
        print("Falling back to local simulation without LLM calls if token is missing.")

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token or "dummy-token"
    )
    
    tasks = ['easy', 'medium', 'hard']
    for t in tasks:
        print(f"\n--- Running Task: {t.upper()} ---")
        env = DriverSafetyEnv(task_level=t)
        state = env.reset()
        
        for step in range(10): # Demo 10 steps per task
            if hf_token:
                action_str = get_action_from_llm(client, model_name, state)
            else:
                action_str = "BEEP" if state.drowsiness_state != "ALERT" else "NONE"
                
            if action_str not in ["BEEP", "VOICE", "ALARM", "BLOCK_IGNITION", "NONE"]:
                action_str = "NONE"
                
            action_obj = Action(action_type=action_str)
            result = env.step(action_obj)
            
            print(f"Step {step+1:02d} | State: {result.state.drowsiness_state}({result.state.ear:.2f}) | "
                  f"Drunk: {result.state.drunk_status} | Action: {action_str:5s} | "
                  f"Reward: {result.reward:5.1f} | TechScore: {result.info['score']}")
            
            state = result.state
            if result.done:
                break

if __name__ == "__main__":
    main()
