import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df_heart = pd.read_csv("./heart_disease_dataset (2).csv")

# Rename columns if necessary
df_heart.rename(columns={
    'blood_pressure': 'BloodPressure',
    'glucose_level': 'GlucoseLevel',
    'physical_activity': 'PhysicalActivity'
}, inplace=True)

# Data Preprocessing
features_to_normalize = ['Age', 'BloodPressure', 'Cholesterol', 'GlucoseLevel', 'BMI', 'PhysicalActivity']
scaler = MinMaxScaler()
existing_features = [col for col in features_to_normalize if col in df_heart.columns]

if existing_features:
    df_heart[existing_features] = scaler.fit_transform(df_heart[existing_features])
else:
    raise KeyError(f"None of the specified features exist in the dataset: {features_to_normalize}")

# Debugging output
print("Existing features for normalization:", existing_features)

# Save the scaler for later use
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Separate features and target variable
# Display available columns for debugging
st.write("Dataset Columns:", df_heart.columns.tolist())

# Ensure the target column exists before dropping
target_column = 'HeartDisease'
if target_column in df_heart.columns:
    X = df_heart.drop(columns=[target_column])  # Features
    y = df_heart[target_column]  # Target
else:
    st.error(f"Target column '{target_column}' not found in dataset. Check column names!")

y = df_heart['HeartDisease']  # Target

# Check class distribution
st.write("Class Distribution:")
st.write(y.value_counts())

# Feature Selection
chi2_selector = SelectKBest(score_func=chi2, k='all').fit(X, y)
mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k='all').fit(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Balanced Class Weights
random_forest_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
rf_grid_search = GridSearchCV(random_forest_model, rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

# Best model selection
best_random_forest_model = rf_grid_search.best_estimator_

# Predictions
rf_preds = best_random_forest_model.predict(X_test)

# Model Evaluation
performance_comparison = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [accuracy_score(y_test, rf_preds)],
    'Precision': [precision_score(y_test, rf_preds)],
    'Recall': [recall_score(y_test, rf_preds)],
    'F1 Score': [f1_score(y_test, rf_preds)]
})
st.write("Model Performance:")
st.write(performance_comparison)

# Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)
st.write("Confusion Matrix:")
st.write(cm)

# Save best model
with open("best_heart_disease_model.pkl", "wb") as model_file:
    pickle.dump(best_random_forest_model, model_file)

# Streamlit App
st.title("Heart Disease Prediction App")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=300, value=200)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=200.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
family_history = st.selectbox("Family History of Heart Disease?", ["Yes", "No"])
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0.0, max_value=20.0, value=2.0)

# Convert categorical inputs to numerical format
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

# Prepare input features
input_features = np.array([[
    age, gender, blood_pressure, cholesterol, glucose, bmi, smoking, alcohol, family_history, physical_activity
]])

# Separate features that need to be scaled
scaled_features = input_features[:, [0, 2, 3, 4, 5, 9]]  # Age, BloodPressure, Cholesterol, GlucoseLevel, BMI, PhysicalActivity
non_scaled_features = input_features[:, [1, 6, 7, 8]]    # Gender, Smoking, Alcohol, FamilyHistory

# Load the scaler and transform scaled features
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
scaled_features = scaler.transform(scaled_features)

# Combine scaled and non-scaled features
input_features_final = np.hstack((scaled_features, non_scaled_features))

# Load model for prediction
with open("best_heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Predict when button is clicked
if st.button("Predict"):
    prediction = model.predict(input_features_final)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.write(f"Prediction: **{result}**")

    # Display predicted probabilities
    probabilities = model.predict_proba(input_features_final)
    st.write("Predicted Probabilities:")
    st.write(f"No Heart Disease: {probabilities[0][0]:.2f}")
    st.write(f"Heart Disease: {probabilities[0][1]:.2f}")
