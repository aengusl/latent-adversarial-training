import time
import os
from huggingface_hub import HfApi
from transformers import AutoModel

# Configuration
WAIT_TIME = 5 * 60 * 60  # 5 hours in seconds
LOCAL_CHECKPOINT_DIR = "/path/to/your/checkpoints"  # Replace with your actual path
HF_USERNAME = "aengusl"
MODEL_NAME_PREFIX = "240921_orpo_twin_checkpoint_"

def wait_with_updates(total_wait_time):
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        remaining_time = total_wait_time - elapsed_time
        
        if remaining_time <= 0:
            print("Wait time complete. Starting upload process.")
            break
        
        hours, remainder = divmod(int(remaining_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        time.sleep(60)  # Update every minute

def upload_checkpoints():
    api = HfApi()
    
    # Get all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(LOCAL_CHECKPOINT_DIR) if d.startswith("checkpoint_")]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_num = checkpoint_dir.split("_")[-1]
        local_path = os.path.join(LOCAL_CHECKPOINT_DIR, checkpoint_dir)
        repo_name = f"{MODEL_NAME_PREFIX}{checkpoint_num}"
        
        print(f"Uploading checkpoint {checkpoint_num}...")
        
        # Create or get the repository
        repo_url = api.create_repo(repo_id=f"{HF_USERNAME}/{repo_name}", exist_ok=True)
        
        # Upload the model
        api.upload_folder(
            folder_path=local_path,
            repo_id=f"{HF_USERNAME}/{repo_name}",
            repo_type="model",
        )
        
        print(f"Checkpoint {checkpoint_num} uploaded successfully.")

if __name__ == "__main__":
    print("Starting wait period...")
    wait_with_updates(WAIT_TIME)
    
    print("Beginning upload process...")
    upload_checkpoints()
    
    print("All checkpoints have been uploaded.")