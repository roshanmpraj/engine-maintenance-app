# for model building and evaluation
import pandas as pd
import numpy as np
import os
import joblib # For model saving
# for experimentation tracking and model logging
import mlflow
import mlflow.sklearn
import xgboost as xgb
# for model training and preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for file paths and Hugging Face repositories
api = HfApi(token=os.getenv("HF_TOKEN"))
OUTPUT_DIR = "/content/Predictive_Maintenance/model_building" # Directory where prep.py saved the splits
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NOTE: Set your Hugging Face repo ID for the model artifact
HF_REPO_ID = "Roshanmpraj/engine_predictive_maintenance_data"       # Dataset Repo ID (used to determine folder structure)
MODEL_REPO_ID = "Roshanmpraj/PredictiveMaintenance-XGBoost-Model" # New repo ID for the model

# Define local file paths
XTRAIN_PATH = os.path.join(OUTPUT_DIR, "Xtrain.csv")
XTEST_PATH = os.path.join(OUTPUT_DIR, "Xtest.csv")
YTRAIN_PATH = os.path.join(OUTPUT_DIR, "ytrain.csv")
YTEST_PATH = os.path.join(OUTPUT_DIR, "ytest.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_xgboost_model.pkl')

# --- MLflow Setup ---
# Set the experiment name
mlflow.set_experiment("Engine_Predictive_Maintenance")

# =============================
# 4.1: Load the train and test data from local files
# =============================
try:
    X_train_final = pd.read_csv(XTRAIN_PATH)
    X_test_final = pd.read_csv(XTEST_PATH)
    y_train_final = pd.read_csv(YTRAIN_PATH).iloc[:, 0] # Assume target is the first (and only) column
    y_test_final = pd.read_csv(YTEST_PATH).iloc[:, 0]

    print("Data successfully loaded from local CSV splits.")

except Exception as e:
    print(f"FATAL: Failed to load splits from local CSV files. Ensure prep.py was run successfully. Error: {e}")
    exit() # Exit the script if data loading fails


# =============================
# 4.2: Define a model and parameters
# =============================
# We use a pipeline for preprocessing (StandardScaler) and the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define parameters for tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.1]
}


# =============================
# 4.3 & 4.4: Tune the model and Log all the tuned parameters
# =============================
print("\n--- Starting Model Tuning and MLflow Tracking ---")

# Start an MLflow run to log all experiments
with mlflow.start_run(run_name="GridSearch_XGBoost_Tuning") as parent_run:

    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1', # Use F1-score as the primary metric for classification
        cv=3,
        verbose=1,
        n_jobs=-1 # Use all available cores
    )

    grid_search.fit(X_train_final, y_train_final)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log the best parameters to the MLflow parent run
    mlflow.log_params(best_params)
    mlflow.log_param("best_scoring_metric", "f1")
    mlflow.log_param("algorithm", "XGBoost")


# =============================
# 4.5: Evaluate the model performance
# =============================
y_pred = best_estimator.predict(X_test_final)
# Predict probabilities for AUC score
y_proba = best_estimator.predict_proba(X_test_final)[:, 1]

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test_final, y_pred),
    "precision": precision_score(y_test_final, y_pred),
    "recall": recall_score(y_test_final, y_pred),
    "f1_score": f1_score(y_test_final, y_pred),
    "roc_auc": roc_auc_score(y_test_final, y_proba)
}

print("\n--- Best Model Evaluation on Test Set ---")
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

# Log final test metrics to the MLflow parent run
# Re-open the run context to ensure metrics are logged to the parent run
with mlflow.start_run(run_id=parent_run.info.run_id):
    for metric, value in metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

# Save the best model locally
joblib.dump(best_estimator, MODEL_PATH)
print(f"\nBest model saved locally: {MODEL_PATH}")

# Log the model artifact to MLflow
with mlflow.start_run(run_id=parent_run.info.run_id):
    mlflow.sklearn.log_model(
        sk_model=best_estimator,
        artifact_path="model_artifact",
        registered_model_name="XGBoostPredictiveMaintenance"
    )
    print("Best model logged to MLflow.")

# =============================
# 4.6: Register the best model in the Hugging Face model hub
# =============================
try:
    # Ensure the model repository exists
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)

    # Upload the serialized model file
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo=os.path.basename(MODEL_PATH), # just the filename
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    print(f"Best model uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}")

except Exception as e:
    print(f"Hugging Face model registration failed. Error: {e}")
