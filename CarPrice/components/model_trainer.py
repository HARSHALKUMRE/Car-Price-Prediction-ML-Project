import os, sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from CarPrice.entity.artifact_entity import (RegressionMetricArtifact,
                                             DataTransformationArtifact,
                                             ModelTrainerArtifact)
from CarPrice.entity.config_entity import ModelTrainerConfig
from CarPrice.entity.estimator import CarPriceModel
from CarPrice.exception import CarPriceException
from CarPrice.logger import logging
from CarPrice.utils.main_utils import load_numpy_array_data, load_object, read_yaml_file, save_object
from neuro_mf import ModelFactory
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )

            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)

            
            mae = mean_absolute_error(y_test, y_pred)  
            mse = mean_squared_error(y_test, y_pred)  
            r2 = r2_score(y_test, y_pred) 
            metric_artifact = RegressionMetricArtifact(mean_absolute_error=mae, mean_squared_error=mse, r2_score=r2)

            return best_model_detail, metric_artifact
        except Exception as e:
            raise CarPriceException(e, sys) from e

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path) 

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            car_price_model = CarPriceModel(preprocessing_object=preprocessing_obj,
            trained_model_object=best_model_detail.best_model)
            logging.info("Created Car Price object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, car_price_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CarPriceException(e, sys) from e