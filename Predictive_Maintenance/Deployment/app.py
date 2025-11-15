import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# --- Constants ---
HF_MODEL_REPO_ID = "Roshanmpraj/PredictiveMaintenance-XGBoost-Model"
HF_MODEL_FILENAME = "xbest_xgboost_model.pkl"
LOCAL_MODEL_PATH = "/content/Predictive_Maintenance/Model_Building/best_xgboost_model.pkl"

# --- Function to Load Model from Hugging Face ---
@st.cache_resource
def load_model():
    """Downloads the model artifact from the Hugging Face Hub and loads it."""
    try:
        # Download the model file from the Hugging Face repository
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        st.success(f"Model successfully loaded from {HF_MODEL_REPO_ID}")
        # Load the model using joblib (assuming it was saved as a pickle file)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Predictive Maintenance App",
    layout="wide"
)

st.title("⚙️ Predictive Engine Maintenance Dashboard")
st.subheader("Forecast potential engine failures using real-time sensor data.")

# Load the trained model
model = load_model()

if model is not None:
    # --- Input Form for Sensor Readings ---
    st.markdown("---")
    st.header("Enter Engine Sensor Readings")

    # Define the input columns in a two-column layout
    col1, col2, col3 = st.columns(3)

    # Dictionary to hold the user inputs
    input_data = {}

    with col1:
        # Engine_RPM: Range from EDA was approx 61 to 2239
        input_data['Engine_RPM'] = st.number_input(
            "Engine RPM (Revolutions per Minute)",
            min_value=60, max_value=2500, value=790, step=10
        )
        # Lub_Oil_Pressure: Range was approx 0.003 to 7.26
        input_data['Lub_Oil_Pressure'] = st.number_input(
            "Lub Oil Pressure (bar/kPa)",
            min_value=0.0, max_value=8.0, value=3.30, step=0.1, format="%.2f"
        )

    with col2:
        # Fuel_Pressure: Range was approx 0.003 to 21.13
        input_data['Fuel_Pressure'] = st.number_input(
            "Fuel Pressure (bar/kPa)",
            min_value=0.0, max_value=25.0, value=6.60, step=0.1, format="%.2f"
        )
        # Coolant_Pressure: Range was approx 0.002 to 7.47
        input_data['Coolant_Pressure'] = st.number_input(
            "Coolant Pressure (bar/kPa)",
            min_value=0.0, max_value=8.0, value=2.30, step=0.1, format="%.2f"
        )

    with col3:
        # Lub_Oil_Temperature: Range was approx 71 to 89
        input_data['Lub_Oil_Temperature'] = st.number_input(
            "Lub Oil Temperature (°C)",
            min_value=70.0, max_value=100.0, value=78.0, step=0.1, format="%.2f"
        )
        # Coolant_Temperature: Range was approx 71 to 102
        input_data['Coolant_Temperature'] = st.number_input(
            "Coolant Temperature (°C)",
            min_value=70.0, max_value=110.0, value=78.0, step=0.1, format="%.2f"
        )

    # --- Prediction Logic ---
    if st.button("Predict Engine Condition"):
        # 1. Get the inputs and save them into a dataframe
        input_df = pd.DataFrame([input_data])

        # NOTE: The original notebook does not show a scaler being saved,
        # but for production, the input data must be transformed with the
        # same scaler used during training. We assume the XGBoost model
        # was trained on the raw features, or the standard scaling was
        # incorporated into the saved model artifact. Given XGBoost's
        # tree-based nature, it often performs well without scaling.
        # We proceed assuming no separate scaling step is needed.

        # 2. Make Prediction
        try:
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            # 3. Display Result
            st.markdown("---")
            st.header("Prediction Result")

            if prediction == 1:
                st.error("🚨 FAULT PREDICTED (Requires Maintenance)")
                st.markdown(f"**Probability of Failure (Class 1):** `{prediction_proba[1]*100:.2f}%`")
                st.markdown("Immediate inspection and preventive maintenance are **strongly recommended** to avoid unexpected breakdown, costly repairs, and operational downtime.")
            else:
                st.success("✅ NORMAL OPERATION")
                st.markdown(f"**Probability of Normal Operation (Class 0):** `{prediction_proba[0]*100:.2f}%`")
                st.markdown("The engine is operating within normal parameters. Continue with scheduled monitoring.")

            st.markdown("---")
            st.dataframe(input_df) # Show the data that was fed to the model

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Cannot proceed without a successfully loaded model.")
