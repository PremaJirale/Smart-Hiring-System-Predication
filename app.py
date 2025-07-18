import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

# Load dataset for dropdown values
@st.cache_data
def load_dataset():
    df = pd.read_csv("artifacts/data.csv") 
    return df

df = load_dataset()

st.set_page_config(page_title="Smart Hiring System", layout="centered")
st.title(" Smart Hiring Prediction System")

# it is the form UI
with st.form("prediction_form"):
    st.subheader("Enter Job and Candidate Details")

    location = st.selectbox("Location", sorted(df['location'].dropna().unique()))
    required_experience = st.selectbox("Required Experience", sorted(df['required_experience'].dropna().unique()))
    required_education = st.selectbox("Required Education", sorted(df['required_education'].dropna().unique()))
    
    job_title = st.selectbox("Job Title", sorted(df['job_title'].dropna().unique()))  
    
    job_description = st.text_area("Job Description")
    job_requirements = st.text_area("Job Requirements")
    resume_text = st.text_area("Resume Text")

    resume_length = st.slider("Resume Length (words)", 10, 5000, 500)

    category = st.selectbox("Category", sorted(df['category'].dropna().unique()))
    job_role = st.selectbox("Job Role", sorted(df['JobRole'].dropna().unique()))
    education = st.selectbox("Candidate's Education", sorted(df['Education'].dropna().unique()))
    gender = st.selectbox("Gender", sorted(df['Gender'].dropna().unique()))
    
    total_working_years = st.slider("Total Working Years", 0, 50, 2)
    monthly_income = st.number_input("Monthly Income", 0, 1000000, step=1000)
    work_life_balance = st.slider("Work Life Balance (1=Bad, 4=Excellent)", 1, 4, 3)

    submitted = st.form_submit_button("Predict")

#  it is the Prediction logic
if submitted:
    try:
        input_data = pd.DataFrame([{
            "location": location,
            "required_experience": required_experience,
            "required_education": required_education,
            "job_title": job_title,   # Added here
            "job_description": job_description,
            "job_requirements": job_requirements,
            "resume_text": resume_text,
            "resume_length": resume_length,
            "category": category,
            "JobRole": job_role,
            "Education": education,
            "Gender": gender,
            "TotalWorkingYears": total_working_years,
            "MonthlyIncome": monthly_income,
            "WorkLifeBalance": work_life_balance
        }])

        st.write("📝 Input Data Sent to Model:", input_data)

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_data)

        st.write("🔍 Raw Model Output:", prediction)

        if prediction["fraud_prediction"] == 1:
            st.error("❌ This job is likely **FRAUDULENT**.")
        else:
            st.success("✅ This job is likely **GENUINE**.")
            if prediction["fit_prediction"] == 1:
                st.info(" Candidate is a **GOOD FIT**.")
            else:
                st.warning("👎 Candidate is **NOT A GOOD FIT**.")

    except Exception as e:
        st.error(f"❗ Prediction Error: {e}")