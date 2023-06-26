import os, sys
from typing import Union

import numpy as np
import pandas as pd
from CarPrice.logger import logging
from CarPrice.exception import CarPriceException
from CarPrice.constant import TARGET_COLUMN, SCHEMA_FILE_PATH
from CarPrice.entity.config_entity import DataTransformationConfig
from CarPrice.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from CarPrice.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders.binary import BinaryEncoder


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CarPriceException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CarPriceException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical, categorical, onehot, binary columns from schema config")

            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]
            onehot_columns = self._schema_config["onehot_columns"]
            binary_columns = self._schema_config["binary_columns"]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            onehot_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False)),
            ])

            binary_pipeline = Pipeline(steps=[
                ('binary', BinaryEncoder()),
                ('scaler', StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("onehot_pipeline", onehot_pipeline, onehot_columns),
                    ("binary_pipeline", binary_pipeline, binary_columns)
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor
        except Exception as e:
            raise CarPriceException(e, sys) from e


    def _outlier_capping(self, col, df):
        logging.info(
            "Entered _outlier_capping method of DataTransformation class"
        )
        try:
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lower_limit), col] = lower_limit

            logging.info(
                "Exited _outlier_capping method of DataTransformation class"
            )

            return df
        except Exception as e:
            raise CarPriceException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                numerical_columns = self._schema_config["numerical_columns"]

                continuous_columns = [feature for feature in numerical_columns if len(train_df[feature].unique())>=25]

                for col in continuous_columns:
                    self._outlier_capping(col=col, df=train_df)

                logging.info(f"Outlier capped in train df")

                for col in continuous_columns:
                    self._outlier_capping(col=col, df=test_df)

                logging.info(f"Outlier capped in test df")

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                logging.info("Got train features and test features of Training dataset")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]
                logging.info("Got train features and test features of Testing dataset")
                
                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                 
                logging.info(f"Data Transformation artifact: {data_transformation_artifact}")
                return data_transformation_artifact
            
            else:
                raise Exception(self.data_validation_artifact.message)


        except Exception as e:
            raise CarPriceException(e, sys) from e

