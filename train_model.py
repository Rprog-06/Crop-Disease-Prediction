# train_model.py

import os
import numpy as np
import joblib
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import extract_features

# Path to dataset (organized as: dataset/class_name/image.jpg)
dataset_path = r"PlantVillage/PlantVillage"


# Collect all image file paths
image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]

if not image_paths:
    raise FileNotFoundError(f"No image files found in '{dataset_path}'. Check dataset path.")

# Get class labels from folder names
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

features = []
labels = []

print("Extracting features...")
for label in classes:
    class_folder = os.path.join(dataset_path, label)
    if not os.path.isdir(class_folder):
        continue
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        feature_vector = extract_features(img_path)
        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(label)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

if len(X) == 0:
    raise ValueError("No valid features extracted. Check dataset folder and image format.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crop_disease_model.pkl")

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
