import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class TrainingPipeline:
    def __init__(self):
        pass

    def initiate_training_pipeline(self):
        try:
           #S logger.info("=== Pipeline started ===")

            # Data Ingestion
           # logger.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
           # logger.info(f"Data ingested. Train: {train_data_path}, Test: {test_data_path}")

            # Data Transformation
            #logger.info("Starting data transformation...")
            transformer = DataTransformation()
            train_arr, test_arr, preprocessor = transformer.initiate_data_transformation(train_data_path, test_data_path)
           # logger.info("Data transformation completed.")

            # Model Training
            logger.info("Starting model training and evaluation...")
            model_trainer = ModelTrainer()
            results = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logger.info("Model training completed.")

            # Save models
            fraud_model = results["fraud"]["model"]
            fit_model = results["fit"]["model"]

            fraud_model_path = os.path.join("artifacts", "fraud_model.pkl")
            fitness_model_path = os.path.join("artifacts", "fitness_model.pkl")

            save_object(fraud_model_path, fraud_model)
            save_object(fitness_model_path, fit_model)

            # ✅ Optional: Save evaluation metrics separately
            save_object("artifacts/fraud_model_metrics.pkl", results["fraud"])
            save_object("artifacts/fitness_model_metrics.pkl", results["fit"])

            logger.info(f"Models saved at: {fraud_model_path}, {fitness_model_path}")

            # ✅ FIXED: Print evaluation results with type-check for formatting
            print("\n=== Model Evaluation Results ===\n")
            for model_type, metrics in results.items():
                print(f"{model_type.capitalize()} Model:")
                for metric_name, value in metrics.items():
                    if metric_name == "model":
                        continue
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name:<20}: {value:.4f}")
                    else:
                        print(f"  {metric_name:<20}: {value}")
                print()

            logger.info("=== Pipeline completed successfully ===")

            return fraud_model_path, fitness_model_path

        except Exception as e:
            logger.error("Pipeline failed.")
            raise CustomException(e, sys)  # ✅ FIXED: Added sys for traceback info


if __name__ == "__main__":
    logger.info("Training pipeline execution started...")
    pipeline = TrainingPipeline()
    fraud_path, fit_path = pipeline.initiate_training_pipeline()
    print(f"Fraud model saved at: {fraud_path}")
    print(f"Fitness model saved at: {fit_path}")
    logger.info("Training pipeline execution completed.")