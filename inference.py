import os
from openai import OpenAI
from server.environment import DriverSafetyEnv, Action, State

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    err_str = error if error is not None else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_str}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

def get_action_from_llm(client: OpenAI, model_name: str, state: State) -> str:
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
        for a in ["BLOCK_IGNITION", "ALARM", "VOICE", "BEEP", "NONE"]:
            if a in action_raw:
                return a
        return "NONE"
    except Exception as e:
        return "NONE"

def fallback_action(task_level: str, state: State) -> str:
    if task_level == 'easy':
        if state.drowsiness_state == 'DROWSY': return 'VOICE'
    elif task_level == 'medium':
        if state.drowsiness_state in ('CRITICAL', 'MICROSLEEP'): return 'ALARM'
        elif state.drowsiness_state == 'DROWSY': return 'VOICE'
    elif task_level == 'hard':
        if state.drunk_status == 'DRUNK': return 'BLOCK_IGNITION'
        elif state.drowsiness_state in ('CRITICAL', 'MICROSLEEP'): return 'ALARM'
        elif state.drowsiness_state == 'DROWSY': return 'VOICE'
    return 'NONE'

def main():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    API_KEY = os.getenv("HF_TOKEN")
    
    # User requested requirement #2
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = ['easy', 'medium', 'hard']
    env_name = "driver-safety"

    for t in tasks_to_run:
        log_start(t, env_name, MODEL_NAME)
        
        env = DriverSafetyEnv(task_level=t)
        state = env.reset()
        
        rewards = []
        final_score = 0.0
        success = False
        steps_taken = 0
        error_msg = None
        
        for step_num in range(1, 21):
            if API_KEY:
                action_str = get_action_from_llm(client, MODEL_NAME, state)
            else:
                action_str = fallback_action(t, state)
                
            if action_str not in ["BEEP", "VOICE", "ALARM", "BLOCK_IGNITION", "NONE"]:
                action_str = "NONE"

            try:
                res = env.step(Action(action_type=action_str))
                state = res.state
                r = float(res.reward)
                d = bool(res.done)
                
                # Fetch score and cap between 0.0 and 1.0
                raw_score = float(res.info.get("score", 0.0))
                final_score = max(0.0, min(1.0, raw_score))
                
                log_step(step_num, action_str, r, d, None)
                rewards.append(r)
                steps_taken = step_num
                
                if d:
                    break
            except Exception as e:
                error_msg = str(e)
                log_step(step_num, action_str, 0.0, True, error_msg)
                steps_taken = step_num
                break

        success = True if error_msg is None else False
        log_end(success, steps_taken, final_score, rewards)

if __name__ == "__main__":
    main()
