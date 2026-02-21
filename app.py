import streamlit as st
import joblib
import numpy as np
import tempfile
import gdown
import os
from feature_extractor import extract_features

MODEL_FILE="crop_disease_model.pkl"
#https://drive.google.com/file/d/1qlW3yEQAUH01Fow-iuVoD8us9MUcMxgf/view?usp=drive_link
# Load model"crop_disease_model.pkl"
if not os.path.exists(MODEL_FILE):
    file_id = "1qlW3yEQAUH01Fow-iuVoD8us9MUcMxgf"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILE, quiet=False)
model = joblib.load(MODEL_FILE)

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
