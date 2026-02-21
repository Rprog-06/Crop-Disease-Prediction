import streamlit as st
import joblib
import numpy as np
import tempfile
from feature_extractor import extract_features

# Load model
model = joblib.load("crop_disease_model.pkl")

# UI
st.title("🌱 Crop Disease Prediction App")
st.write("Upload a crop leaf image to predict disease type.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Extract features
    features = extract_features(temp_path).reshape(1, -1)

    # Prediction
    prediction = model.predict(features)[0]
    st.success(f"Prediction: {prediction}")
