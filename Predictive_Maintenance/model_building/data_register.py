from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# --- 1. Configuration ---
# Replace 'YOUR_HF_USERNAME' with your actual Hugging Face username
HF_USERNAME = "Roshanmpraj"
repo_id = f"{HF_USERNAME}/engine_predictive_maintenance_data"
repo_type = "dataset"
data_folder_path = "Predictive_Maintenance/data"

# --- 2. Check for Token and Initialize API ---
# This explicit check ensures the user is aware of the missing token,
# which was the cause of the 401 Unauthorized error.
if "HF_TOKEN" not in os.environ:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please set it to your Hugging Face write token (with 'write' role) before running. "
        "Example in terminal: export HF_TOKEN='hf_...'"
    )

# Initialize API client (It will automatically use the HF_TOKEN from the environment)
api = HfApi()

# --- 3. Execution ---

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
# This operation requires the HF_TOKEN to have write access.
api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Contents of '{data_folder_path}' uploaded to Hugging Face successfully.")
