# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for clean exit on error
import sys
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# --- Constants & Configuration ---
# NOTE: Update these with your actual repo IDs
DATASET_REPO_ID = "Roshanmpraj/engine_predictive_maintenance_data" # Corrected repository ID
MODEL_REPO_ID = "Roshanmpraj/PredictiveMaintenance-XGBoost-Model"

# =============================
# MLflow setup
# =============================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Predictive-Maintenance-Training")

# =============================
# Hugging Face setup
# =============================
api = HfApi(token=os.getenv("HF_TOKEN"))

# =============================
# Download dataset splits from Hugging Face
# This assumes Criteria 3 (data prep) has been run and splits uploaded.
# =============================
try:
    print(f"Downloading data splits from {DATASET_REPO_ID}...")
    Xtrain_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtrain.csv", repo_type="dataset")
    Xtest_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtest.csv", repo_type="dataset")
    ytrain_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytrain.csv", repo_type="dataset")
    ytest_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytest.csv", repo_type="dataset")
    print("Download complete.")

    Xtrain = pd.read_csv(Xtrain_path)
    Xtest = pd.read_csv(Xtest_path)
    # Ensure target is a Series
    ytrain = pd.read_csv(ytrain_path).iloc[:, 0].squeeze()
    ytest = pd.read_csv(ytest_path).iloc[:, 0].squeeze()

except Exception as e:
    # This block handles the HTTPError/RepositoryNotFoundError
    print(f"Error downloading or loading data splits. Ensure the repo ID is correct and the files exist. Error: {e}")
    # Use sys.exit() instead of exit()
    sys.exit(1)

# =============================
# Features (all are numeric based on engine_data.csv)
# =============================
numeric_features = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure',
    'Lub_Oil_Temperature', 'Coolant_Temperature'
]

# Ensure the columns match the expected features
Xtrain = Xtrain[numeric_features]
Xtest = Xtest[numeric_features]


# =============================
# Class imbalance handling
# =============================
# Calculate the ratio of the majority class (0) to the minority class (1)
# This is a common way to handle binary classification imbalance with XGBoost
if 1 in ytrain.value_counts().index:
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    print(f"Class Imbalance Ratio (0/1): {class_weight:.2f}")
else:
    class_weight = 1.0
    print("Warning: Only one class found in ytrain. Setting class_weight to 1.0.")


# Preprocessor: Only standard scaling needed for numeric data
preprocessor = StandardScaler()

# Base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, # Apply class weight
    random_state=42,
    use_label_encoder=False,        # Suppress deprecation warning
    eval_metric='logloss'           # Appropriate metric for binary classification
)

# Model pipeline: Preprocessor (Scaler) + Model
# Note: For single-step preprocessing like StandardScaler, make_pipeline is cleaner
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05],
    "xgbclassifier__subsample": [0.7, 0.9],
}

# =============================
# Training with MLflow logging
# =============================
print("\n--- Starting Model Training (Grid Search) and MLflow Logging ---")
with mlflow.start_run():
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1' # Prioritize F1-score as this is predictive maintenance (Recall is also critical)
    )
    grid_search.fit(Xtrain, ytrain)

    # Log only the best model params
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("classification_threshold", 0.5) # Log default threshold

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions (using default threshold 0.5 for metrics)
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics (Focus on class 1 - Engine Condition = 1)
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_recall_fail": train_report["1"]["recall"], # Recall for the failure class
        "train_f1_fail": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_recall_fail": test_report["1"]["recall"],
        "test_f1_fail": test_report["1"]["f1-score"],
    })
    print("\nMLflow Metrics Logged:")
    print(f"Test F1 (Failure Class): {test_report['1']['f1-score']:.4f}")

    # Save the model locally
    model_path = "best_maintenance_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Best model saved locally at: {model_path}")

    # Log the model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(__file__, artifact_path="scripts") # Log the training script itself
    print("Model and script logged as MLflow artifacts.")

    # =============================
    # Register Model to Hugging Face Model Hub
    # =============================
    print("\n--- Registering Model to Hugging Face Model Hub ---")

    try:
        # Check if the model repo exists, create if not
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Model Repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model Repo '{MODEL_REPO_ID}' not found. Creating new repo...")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
        print(f"Repo '{MODEL_REPO_ID}' created.")

    # Upload the serialized model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path), # File name in the repo
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"Best model uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}/{os.path.basename(model_path)}")

print("\nTraining and Model Registration complete.")
