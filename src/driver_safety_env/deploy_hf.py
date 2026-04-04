import os
from huggingface_hub import HfApi

def deploy():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        return
        
    api = HfApi(token=token)
    
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Authenticated as: {username}")
        
        repo_id = f"{username}/driver-safety-env"
        print(f"Targeting Space: {repo_id}")
        
        # Add HF_TOKEN as secret correctly
        try:
            api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
            print("Successfully injected HF_TOKEN into Space configuration!")
        except Exception as e:
            print(f"Failed to set HF_TOKEN secret: {e}")
        
        try:
            api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
            print(f"Space ready: {repo_id}")
        except Exception as e:
            print(f"Failed to create repo (might already exist): {e}")
            
        print("Uploading files...")
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["__pycache__/*", ".git/*", ".gemini/*", "deploy_hf.py"],
            commit_message="Initial deployment"
        )
        print(f"Success! Space is building at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy()
