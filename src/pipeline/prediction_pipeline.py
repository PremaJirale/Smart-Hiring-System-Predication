import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger import logging


class PredictionPipeline:
    def __init__(self):
        try:
            self.fraud_model_path = os.path.join('models', 'fraud_model.pkl')
            self.fitness_model_path = os.path.join('models', 'fitness_model.pkl')
            self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # Load fraud model
            with open(self.fraud_model_path, 'rb') as f:
                self.fraud_model = pickle.load(f)
            logging.info("Fraud model loaded successfully.")

            # Load fitness model
            with open(self.fitness_model_path, 'rb') as f:
                self.fitness_model = pickle.load(f)
            logging.info("Fitness model loaded successfully.")

            # Load preprocessor
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logging.info("Preprocessor loaded successfully.")

        except Exception as e:
            logging.error("Error loading models or preprocessor.")
            raise CustomException(e, sys)

    def predict(self, input_data):
        try:
            # Convert input to DataFrame if dict
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            elif not isinstance(input_data, pd.DataFrame):
                raise ValueError("Input data must be a dictionary or pandas DataFrame.")

            required_columns = [
                'job_title', 'location', 'required_experience', 'required_education',
                'job_description', 'job_requirements', 'resume_text', 'resume_length',
                'category', 'JobRole', 'Education', 'Gender', 'TotalWorkingYears',
                'MonthlyIncome', 'WorkLifeBalance'
            ]

            # Check for missing columns
            missing_cols = set(required_columns) - set(input_data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            logging.info("Input data received for prediction.")

            # Transform input using preprocessor
            transformed_data = self.preprocessor.transform(input_data)
            logging.info("Data transformed successfully.")

            # Predict fraud
            fraud_preds = self.fraud_model.predict(transformed_data)
            logging.info(f"Fraud predictions: {fraud_preds}")

            results = []

            for i, fraud in enumerate(fraud_preds):
                if fraud == 1:
                    # Job is predicted as fraud
                    results.append({
                        "fraud_prediction": 1,
                        "fit_prediction": None,
                        "message": "🚨 This job is likely FRAUDULENT."
                    })
                else:
                    # Job is genuine, check candidate fitness
                    fit_pred = self.fitness_model.predict([transformed_data[i]])[0]
                    logging.info(f"Fitness prediction: {fit_pred}")

                    fit_message = "👍 Candidate is a GOOD FIT." if fit_pred == 1 else "👎 Candidate is NOT A GOOD FIT."

                    results.append({
                        "fraud_prediction": 0,
                        "fit_prediction": int(fit_pred),
                       # "message": f"✅ This job is likely GENUINE.\n{fit_message}"
                    })

            return results[0] if len(results) == 1 else results

        except Exception as e:
            logging.error("Error occurred during prediction.")
            raise CustomException(e, sys) 