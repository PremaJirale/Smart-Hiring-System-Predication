<<<<<<< HEAD
# Smart Hiring System #

End-to-end Machine Learning pipeline that predicts job fraudulence and evaluates candidate-job fitness to support intelligent hiring decisions.


## Project Overview ##
The Smart Hiring System uses machine learning to solve two key problems:

Fraud Job Detection: Identifies whether a job posting is genuine or fraudulent.

Candidate-Job Fitness Evaluation: Predicts whether a candidate is suitable for the job, but only if the job is legitimate.

This intelligent flow helps protect job seekers from scams and assists employers in selecting the best candidates.

# How It Works #
 
 Two-Step Prediction:
Step 1 – Fraud Detection (label_fraud)

Input: Job details (title, location, experience, education, description, etc.)

Output: 1 for fraudulent job, 0 for genuine.

Step 2 – Fitness Evaluation (label_fit)

Only if the job is not fraudulent.

Input: Candidate's resume and job requirements.

Output: 1 if the candidate fits the job, 0 otherwise

# Project Structure #

PremaJirale/
├── models/ → Trained ML models (.pkl)
├── artifacts/ → Train/test data & preprocessor
├── src/
│ └── components/
│ ├── data_ingestion.py
│ ├── data_transformation.py
│ ├── model_trainer.py
│ └── pipeline/
│ ├── train_pipeline.py
│ └── prediction_pipeline.py
│ ├── utils.py
│ ├── logger.py
│ └── exception.py
├── app.py → Streamlit App
├── requirements.txt
└── README.md

###  Technologies Used
- Python
- Streamlit (for UI)
- Random Forest (for Fraud Detection)
- Decision Tree (for Fit Prediction)
- Sklearn, Pandas, NumPy,seaborn,imbalnce-learn
- Pickle for saving models

### Project Workflow
data_ingestion → data_transformation → model_trainer → prediction_pipeline → streamlit UI

# How to Run #

1️⃣ Set up environment
git clone <https://github.com/PremaJirale/Smart-Hiring-System-predication
cd PremaJirale
python -m venv venv
venv\Scripts\activate 
pip install -r requirements.txt

2️⃣ Train the model
python -m src.pipeline.train_pipeliene

3️⃣ Run the App
 streamlit run app.py


 # sample output #

"fraud_prediction":0
"fit_prediction":0

✅ This job is likely GENUINE.

👎 Candidate is NOT A GOOD FIT.
=======
# Smart-Hiring-System-predication
End-to-end Machine Learning pipeline that predicts job fraudulence and evaluates candidate-job fitness to support intelligent hiring decisions.
>>>>>>> af0563739c71cb530443ac788469f8f371befb4f
