# for data manipulation
import pandas as pd
import numpy as np
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for feature scaling (Crucial for predictive maintenance sensor data)
from sklearn.preprocessing import StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
OUTPUT_DIR = "Predictive_Maintenance/Model_Building"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setting dataset path
DATASET_PATH = "/content/Predictive_Maintenance/data/engine_data.csv" # Using the uploaded filename for local context

# NOTE: Set your Hugging Face repo ID for upload
HF_REPO_ID = "Roshanmpraj/engine_predictive_maintenance_data"


# =============================
# 1. Load Dataset
# =============================
try:
    df = pd.read_csv(DATASET_PATH)
    # Standardize column names based on your initial file structure (camelcase to snake_case for consistency)
    df.columns = [
        'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure',
        'Lub_Oil_Temperature', 'Coolant_Temperature', 'Engine_Condition'
    ]
    print("Dataset loaded and columns standardized successfully.")
except Exception as e:
    print(f"Failed to load dataset from {DATASET_PATH}. Error: {e}")


# =============================
# 2. Data Cleaning & Feature Engineering
# =============================

# Ensure target variable is integer type
df['Engine_Condition'] = df['Engine_Condition'].astype(int)
print("Data types ensured for modeling.")


# =============================
# 3. Define target + features and Split
# =============================
target_col = "Engine_Condition"
# All remaining columns are features except the target
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split (stratifying is critical for classification tasks)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 4. Save Locally
# =============================
Xtrain.to_csv(os.path.join(OUTPUT_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(OUTPUT_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(OUTPUT_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(OUTPUT_DIR, "ytest.csv"), index=False)
print(f"Train/Test splits saved locally in: {OUTPUT_DIR}")


# =============================
# 5. Upload to Hugging Face
# =============================
files_to_upload = [
    os.path.join(OUTPUT_DIR, "Xtrain.csv"),
    os.path.join(OUTPUT_DIR, "Xtest.csv"),
    os.path.join(OUTPUT_DIR, "ytrain.csv"),
    os.path.join(OUTPUT_DIR, "ytest.csv"),
]

for file_path in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"Successfully uploaded: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Failed to upload {os.path.basename(file_path)}. Error: {e}")
