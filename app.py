import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_and_resources():
    model = load_model("exam_score_model.keras", compile=False)
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_resources()

levels = ['Low', 'Medium', 'High']

# User Interface
st.title("📝 Student Exam Score Prediction")

attendance = st.number_input(
    "Attendance of student ",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

hours_studied = st.number_input(
    "How much hours student study",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

previous_scores = st.number_input(
    "Score for previous exam",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

tutoring_sessions = st.number_input(
    "Tutoring sessions",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

access_to_resources = st.selectbox(
    "Does student have access to resource",
    levels
)

parental_involvement = st.selectbox(
    "Parental involvement",
    levels
)

motivation_level = st.selectbox(
    "Motivation level of student",
    levels
)

internet_access = st.radio(
    "Does student have access to internet_Access",
    ["Yes", "No"]
)

family_income = st.selectbox(
    "Family income",
    levels
)

teacher_quality = st.selectbox(
    "Teacher quality ",
    levels
)

peer_influence = st.radio(
    "Peer influence on a student",
    ["Positive", "Negative"]
)

#Preprocessing
def preprocess_input(attendance, hours_studied, previous_scores, tutoring_sessions, access_to_resources, parental_involvement, motivation_level, internet_access, family_income, teacher_quality, peer_influence):
    df = pd.DataFrame({
        'Attendance': [attendance],
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Tutoring_Sessions': [tutoring_sessions],
        'Access_to_Resources': [access_to_resources],
        'Parental_Involvement': [parental_involvement],
        'Motivation_Level': [motivation_level],
        'Internet_Access': [internet_access],
        'Family_Income': [family_income],
        'Teacher_Quality': [teacher_quality],
        'Peer_Influence': [peer_influence]
    })

    df_processed = preprocessor.transform(df)

    return df_processed

# Prediction
if st.button("Predict Exam Score"):
    input_data = preprocess_input(attendance, hours_studied, previous_scores, tutoring_sessions, access_to_resources, parental_involvement, motivation_level, internet_access, family_income, teacher_quality, peer_influence)
    pred = model.predict(input_data, verbose=0)
    st.success(f"Estimated exam score: ${pred[0][0]:,.2f}")