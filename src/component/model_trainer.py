import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class ModelTrainer:
    def __init__(self):
        # Define directory to store models
        self.model_dir = "models"
        self.fraud_model_path = os.path.join(self.model_dir, "fraud_model.pkl")
        self.fitness_model_path = os.path.join(self.model_dir, "fitness_model.pkl")
        os.makedirs(self.model_dir, exist_ok=True)

        # Define models
        self.fraud_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42)
        }

        self.fitness_models = {
            "DecisionTree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42)
        }

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return {
            "Accuracy": accuracy,
            "Classification_Report": report
        }

    def train_and_evaluate_models(self, models_dict, X_train, y_train, X_test, y_test):
        results = {}
        accuracies = {}
        trained_models = {}

        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
            accuracies[name] = metrics["Accuracy"]
            trained_models[name] = model
            logger.info(f"{name} Accuracy: {metrics['Accuracy']:.4f}")
            logger.info(f"{name} Report:\n{metrics['Classification_Report']}")

        # Select the best model
        best_model_name = max(accuracies, key=accuracies.get)
        best_model = trained_models[best_model_name]
        best_accuracy = accuracies[best_model_name]
        logger.info(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

        results["Best_Model"] = {
            "Model_Name": best_model_name,
            "Accuracy": best_accuracy,
            "Model_Object": best_model
        }

        return results

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting features and targets from train and test arrays...")

            X_train = train_array[:, :-2]
            y_train_fraud = train_array[:, -2]
            y_train_fit = train_array[:, -1]

            X_test = test_array[:, :-2]
            y_test_fraud = test_array[:, -2]
            y_test_fit = test_array[:, -1]

            # Train and evaluate fraud detection models
            logger.info("Training fraud detection models...")
            fraud_results = self.train_and_evaluate_models(self.fraud_models, X_train, y_train_fraud, X_test, y_test_fraud)
            fraud_best_model = fraud_results["Best_Model"]["Model_Object"]
            save_object(self.fraud_model_path, fraud_best_model)

            # Train and evaluate fitness prediction models
            logger.info("Training fitness detection models...")
            fitness_results = self.train_and_evaluate_models(self.fitness_models, X_train, y_train_fit, X_test, y_test_fit)
            fitness_best_model = fitness_results["Best_Model"]["Model_Object"]
            save_object(self.fitness_model_path, fitness_best_model)

            logger.info("Both models trained and saved successfully.")

            # Return evaluation summary
            return {
                "fraud": {
                    "model": fraud_best_model,
                    "model_name": fraud_results["Best_Model"]["Model_Name"],
                    "accuracy": fraud_results["Best_Model"]["Accuracy"]
                },
                "fit": {
                    "model": fitness_best_model,
                    "model_name": fitness_results["Best_Model"]["Model_Name"],
                    "accuracy": fitness_results["Best_Model"]["Accuracy"]
                }
            }

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys) 