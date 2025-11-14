from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# --- 1. Configuration (From First Option) ---
# Replace 'YOUR_HF_USERNAME' with your actual Hugging Face username
HF_USERNAME = "Roshanmpraj"
repo_id = f"{HF_USERNAME}/engine_predictive_maintenance_data" # Use a dynamic ID
repo_type = "dataset"
data_folder_path = "/content/Predictive_Maintenance/data" # Use the folder you created in Step 1.1

# --- 2. Execution (From Second Option) ---
# Initialize API client (uses HF_TOKEN environment variable if not specified)
api = HfApi()

# Step 1: Check if the space exists (Robustness)
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    # NOTE: Set private=True if you do not want your dataset public
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload the data folder (Efficiency)
api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Contents of '{data_folder_path}' uploaded to Hugging Face.")
