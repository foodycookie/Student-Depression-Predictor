import joblib
import pandas as pd
import streamlit as st

# Load feature columns
feature_columns = joblib.load('model/feature_columns.pkl')

# Load models
decision_tree_model = joblib.load('model/Decision Tree.pkl')
random_forest_model = joblib.load('model/Random Forest.pkl')
xgboost_model = joblib.load('model/XGBoost.pkl')

df = pd.read_csv("Student Depression Dataset.csv")

# ----------------------------------------------------------------------------------------------------------------------------------

# UI
st.title("Student Depression Predictor")

with st.form("input_form"):
    gender = st.radio("Gender", options=['Male', 'Female'])
    
    age = st.number_input("Age", min_value=18, max_value=59, value=21, step=1)
    
    profession = st.selectbox("Profession", options=sorted(df['Profession'].unique()))
    
    academic_pressure = st.number_input("Academic Pressure", min_value=0, max_value=5, value=3, step=1)
    
    work_pressure = st.number_input("Work Pressure", min_value=0, max_value=5, value=3, step=1)
    
    cgpa = st.number_input("CGPA", min_value=0.00, max_value=10.00, value=5.00, step=0.1)
    
    study_satisfaction = st.number_input("Study Satisfaction", min_value=0, max_value=5, value=3, step=1)
    
    job_satisfaction = st.number_input("Job Satisfaction", min_value=0, max_value=5, value=3, step=1)
    
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])
    
    dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy", "Others"])
    
    degree_level = st.selectbox("Degree Level", ["School", "Bachelor", "Master", "Doctorate", "Others"])
    
    suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
    
    work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=12, value=6, step=1)
    
    financial_stress = st.number_input("Financial Stress", min_value=0, max_value=5, value=3, step=1)
    
    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predict")

# Prepare input for model
if submitted:
    profession_df = pd.get_dummies(pd.DataFrame([profession]), prefix="Profession")
    
    sleep_duration_map  = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3, "Others": 1}
    dietary_habits_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2, "Others": 1}
    degree_level_map = {"School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3, "Others": 4}
    
    gender_map = {'Male': 1, 'Female': 0}
    suicidal_thoughts_map = {'Yes': 1, 'No': 0}
    family_history_map = {'Yes': 1, 'No': 0}
    
    # Combine into dataframe
    input_df = pd.DataFrame({
        "Gender_Male": [gender_map[gender]],
        "Age": [age],
        "Academic Pressure": [academic_pressure],
        "Work Pressure": [work_pressure],
        "CGPA": [cgpa],
        "Study Satisfaction": [study_satisfaction],
        "Job Satisfaction": [job_satisfaction],
        "Sleep Duration": [sleep_duration_map[sleep_duration]],
        "Dietary Habits": [dietary_habits_map[dietary_habits]],
        "Degree Level": [degree_level_map[degree_level]],
        "Have you ever had suicidal thoughts ?": [suicidal_thoughts_map[suicidal_thoughts]],
        "Work/Study Hours": [work_study_hours],
        "Financial Stress": [financial_stress],
        "Family History of Mental Illness": [family_history_map[family_history]]
    })
    
    input_df = pd.concat([input_df, profession_df], axis=1)
    
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    models = {
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model
    }
    
    st.subheader("Model Comparison Summary")
    
    results = []
    
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        results.append({
            "Model": name,
            "Prediction": "Depression" if pred == 1 else "No Depression",
            "Confidence": round(prob if pred == 1 else 1 - prob, 2)
        })
        
    st.dataframe(results)