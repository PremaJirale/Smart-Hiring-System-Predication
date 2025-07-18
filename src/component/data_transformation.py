import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "resume_length", 
                "TotalWorkingYears", 
                "MonthlyIncome", 
                "WorkLifeBalance"
            ]

            categorical_columns = [
                "category", 
                "Education", 
                "Gender", 
                "required_experience", 
                "required_education", 
                "JobRole"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(
                    drop='first', 
                    handle_unknown='ignore',
                    sparse_output=False  # Ensure dense output for compatibility
                )),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name_fraud = "label_fraud"
            target_column_name_fit = "label_fit"

            # Drop target columns from features
            input_feature_train_df = train_df.drop(
                columns=[target_column_name_fraud, target_column_name_fit], 
                axis=1
            )
            input_feature_test_df = test_df.drop(
                columns=[target_column_name_fraud, target_column_name_fit], 
                axis=1
            )

            # Extract target columns, reshape for concatenation
            target_feature_train_fraud = train_df[target_column_name_fraud].values.reshape(-1, 1)
            target_feature_train_fit = train_df[target_column_name_fit].values.reshape(-1, 1)
            target_feature_test_fraud = test_df[target_column_name_fraud].values.reshape(-1, 1)
            target_feature_test_fit = test_df[target_column_name_fit].values.reshape(-1, 1)

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Ensure feature arrays are 2D
            if input_feature_train_arr.ndim == 1:
                input_feature_train_arr = input_feature_train_arr.reshape(-1, 1)
            if input_feature_test_arr.ndim == 1:
                input_feature_test_arr = input_feature_test_arr.reshape(-1, 1)

            # Concatenate features and targets for train and test
            train_arr = np.column_stack((
                input_feature_train_arr,
                target_feature_train_fraud,
                target_feature_train_fit
            ))

            test_arr = np.column_stack((
                input_feature_test_arr,
                target_feature_test_fraud,
                target_feature_test_fit
            ))

            # Save the preprocessor object
            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_path), 
                exist_ok=True
            )
            with open(self.data_transformation_config.preprocessor_obj_path, "wb") as file_obj:
                pickle.dump(preprocessing_obj, file_obj)

            logging.info(f"Saved preprocessing object successfully at {self.data_transformation_config.preprocessor_obj_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            error_message = f"Error occurred in file {file_name} at line {line_number}: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)
