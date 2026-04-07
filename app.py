import joblib
import pandas as pd
import streamlit as st

# Load maps
city_map = joblib.load('model/city_map.joblib')
degree_map = joblib.load('model/degree_map.joblib')

# Load feature columns
feature_columns = joblib.load('model/feature_columns.joblib')

# Load scaler
scaler = joblib.load('model/scaler.joblib')

# Load models
ann = joblib.load('model/ann_model.joblib')
knn = joblib.load('model/knn_model.joblib')
svm  = joblib.load('model/svm_model.joblib')

# ──────────────────────────────────────────────────────────────────────────────────────────
# UI
st.title("Student Depression Predictor")

model_choice = st.selectbox("Choose Model", ["ANN", "KNN", "SVM"])

with st.form("input_form"):
    gender = st.radio("Gender", options=['Male', 'Female'])
    
    age = st.number_input("Age", min_value=18, max_value=100, value=18, step=1)
    
    city = st.selectbox("City", options=list(city_map.keys()))
    
    academic_pressure = st.number_input("Academic Pressure", min_value=0, max_value=5, value=3, step=1)
    
    cgpa = st.number_input("CGPA", min_value=0.00, max_value=10.00, value=5.00, step=0.01)
    
    study_satisfaction = st.number_input("Study Satisfaction", min_value=0, max_value=5, value=3, step=1)
    
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])
    
    dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy", "Others"])
    
    degree = st.selectbox("Degree", options=list(degree_map.keys()))
    
    suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
    
    work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=12, step=1)
    
    financial_stress = st.number_input("Financial Stress", min_value=0, max_value=5, value=3, step=1)
    
    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predict")

# Prepare input for model
if submitted:
    gender_male = 1 if gender == "Male" else 0
    
    city_encoded = city_map[city]
    
    sleep_duration_map = {
        'Less than 5 hours': 0,
        '5-6 hours'        : 1,
        '7-8 hours'        : 2,
        'More than 8 hours': 3,
        'Others'           : 1
    }
    
    dietary_habits_map = {
        'Unhealthy': 0,
        'Moderate' : 1,
        'Healthy'  : 2,
        'Others'   : 1
    }
    
    degree_encoded = degree_map[degree]
    
    suicidal_thoughts_map = {'Yes': 1, 'No': 0}
    
    family_history_map = {'Yes': 1, 'No': 0}
    
    # Combine into dataframe
    input_df = pd.DataFrame({
        "Gender_Male": [gender_male],
        "Age": [age],
        "City": [city_encoded],
        "Academic Pressure": [academic_pressure],
        "CGPA": [cgpa],
        "Study Satisfaction": [study_satisfaction],
        "Sleep Duration": [sleep_duration_map[sleep_duration]],
        "Dietary Habits": [dietary_habits_map[dietary_habits]],
        "Degree": [degree_encoded],
        "Have you ever had suicidal thoughts ?": [suicidal_thoughts_map[suicidal_thoughts]],
        "Work/Study Hours": [work_study_hours],
        "Financial Stress": [financial_stress],
        "Family History of Mental Illness": [family_history_map[family_history]]
    })
    
    input_df = input_df[feature_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    if model_choice == "ANN":
        prediction = ann.predict(input_scaled)[0]
        prediction_probability = ann.predict_proba(input_scaled)[0][1]

    elif model_choice == "KNN":
        prediction = knn.predict(input_scaled)[0]
        prediction_probability = knn.predict_proba(input_scaled)[0][1]

    elif model_choice == "SVM":
        prediction = svm.predict(input_scaled)[0]
        prediction_probability = svm.predict_proba(input_scaled)[0][1] 
            
    # Show results
    if prediction == 1:
        st.error(f"Depression! (Confidence: {prediction_probability:.2f})")
    else:
        st.success(f" No Depression! (Confidence: {1-prediction_probability:.2f})")