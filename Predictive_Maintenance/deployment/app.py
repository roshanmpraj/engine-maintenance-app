import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Create the deployment directory if it doesn't exist
os.makedirs("Predictive_Maintenance/deployment", exist_ok=True)

# --- Constants & Configuration ---
MODEL_REPO_ID = "Roshanmpraj/PredictiveMaintenance-XGBoost-Model"
MODEL_FILENAME = "best_maintenance_model.joblib"

# The exact list of numeric features used for training and inference
# MUST match the feature list from your train.py script
NUMERIC_FEATURES = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure',
    'Lub_Oil_Temperature', 'Coolant_Temperature'
]

# =============================
# Streamlit UI Configuration (MUST BE FIRST COMMAND)
# =============================
st.set_page_config(page_title="Engine Condition Prediction", layout="wide")
st.title("⚙️ Engine Condition Predictor (Predictive Maintenance)")
st.markdown("""
This application predicts the likelihood of an **Engine Failure (Condition=1)** based on real-time sensor readings.
The model uses an XGBoost Classifier trained on a scikit-learn pipeline.
---
""")

# =============================
# Load the trained model Pipeline
# =============================
@st.cache_resource
def load_maintenance_model():
    """Downloads and loads the trained joblib pipeline from Hugging Face Hub."""
    try:
        # Download the model artifact
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        # Load the pipeline object
        loaded_pipeline = joblib.load(model_path)
        st.success(f"Model Pipeline loaded successfully from Hugging Face Hub: {MODEL_REPO_ID}")

        # The loaded_pipeline is a scikit-learn Pipeline (Scaler + XGBoost)
        return loaded_pipeline
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.info("Please ensure the model artifact is uploaded correctly to the Hugging Face Hub.")
        st.stop()

model_pipeline = load_maintenance_model()

# =============================
# Streamlit Input UI
# =============================
st.header("Enter Engine Sensor Readings")

# Layout for inputs using three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Pressure Readings")
    lub_oil_pressure = st.slider("Lub Oil Pressure (bar)", 0.0, 10.0, 5.0, 0.1)
    fuel_pressure = st.slider("Fuel Pressure (bar)", 0.0, 10.0, 4.0, 0.1)
    coolant_pressure = st.slider("Coolant Pressure (bar)", 0.0, 10.0, 6.0, 0.1)

with col2:
    st.subheader("Temperature Readings")
    lub_oil_temp = st.slider("Lub Oil Temp (°C)", 0.0, 150.0, 75.0, 1.0)
    coolant_temp = st.slider("Coolant Temp (°C)", 0.0, 150.0, 90.0, 1.0)

with col3:
    st.subheader("Speed")
    engine_rpm = st.number_input("Engine rpm (RPM)", min_value=100, max_value=10000, value=2500, step=100)

# Assemble input data into a DataFrame with the exact feature order
input_data = pd.DataFrame([[
    engine_rpm,
    lub_oil_pressure,
    fuel_pressure,
    coolant_pressure,
    lub_oil_temp,
    coolant_temp
]], columns=NUMERIC_FEATURES)

# -----------------------------
# Prediction Logic
# -----------------------------
st.markdown("---")
if st.button("Predict Engine Condition", type="primary"):
    if model_pipeline:
        try:
            # The loaded pipeline automatically handles scaling and prediction
            # The input_data MUST have the correct features in the correct order (NUMERIC_FEATURES)
            prediction = model_pipeline.predict(input_data)[0]
            # Get probability for class 1 (Failure)
            prediction_proba = model_pipeline.predict_proba(input_data)[0][1] * 100

            st.subheader("Prediction Result:")

            if prediction == 1:
                st.error(f"**🔴 HIGH RISK OF FAILURE (Predicted: {prediction_proba:.2f}% Probability)**")
                st.markdown("Immediate inspection and maintenance are recommended for this engine profile.")
            else:
                st.success(f"**🟢 Engine Condition is Normal (Predicted: {prediction_proba:.2f}% Probability of Failure)**")
                st.markdown("The current sensor readings indicate normal operating condition.")

            st.markdown(f"---")
            st.code(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction. Ensure the loaded model pipeline is valid and data types are correct. Error: {e}")
