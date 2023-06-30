import os
import sys
import logging

from CarPrice.constant import SCHEMA_FILE_PATH
from CarPrice.entity.config_entity import CarPricePredictionConfig
from CarPrice.entity.s3_estimator import CarEstimator
from CarPrice.exception import CarPriceException
from CarPrice.logger import logging
from CarPrice.utils.main_utils import read_yaml_file
import pandas as pd
from pandas import DataFrame

class CarData:
    def __init__(self, car_name: str,
                vehicle_age: int,
                km_driven: int,
                seller_type: str,
                fuel_type: str,
                transmission_type : str,
                mileage : float,
                engine : int,
                max_power : float,
                seats : int,
                selling_price : int = None 
                ):
        """
        Heart Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.car_name = car_name
            self.vehicle_age = vehicle_age
            self.km_driven = km_driven
            self.seller_type = seller_type
            self.fuel_type = fuel_type
            self.transmission_type = transmission_type
            self.mileage = mileage
            self.engine = engine
            self.max_power = max_power
            self.seats = seats
            self.selling_price = selling_price

        except Exception as e:
            raise CarPriceException(e, sys) from e

    def get_car_price_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from HeartData class input
        """
        try:
            
            car_price_input_dict = self.get_car_price_data_as_dict()
            return DataFrame(car_price_input_dict)
        
        except Exception as e:
            raise CarPriceException(e, sys) from e

    def get_car_price_data_as_dict(self):
        """
        This function returns a dictionary from CarData class input 
        """
        try:
            input_data = {
                "car_name": [self.car_name],
                "vehicle_age": [self.vehicle_age],
                "km_driven": [self.km_driven],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats]
                }
            input_data = pd.DataFrame(input_data)
            return input_data
        except Exception as e:
            raise CarPriceException(e, sys) from e


class CarPriceRegressor:
    def __init__(self, prediction_pipeline_config: CarPricePredictionConfig = CarPricePredictionConfig(),) -> None:
        """
        :param prediction_pipeline_config:
        """
        try:
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config 
        except Exception as e:
            raise CarPriceException(e, sys) from e

    
    def predict(self, dataframe) -> str:
        """
        This is the method of HeartStrokeClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info(f"Entered predict method of CarPriceRegressor class")
            model = CarEstimator(
                bucket_name = self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            car_price = model.predict(dataframe)

            return car_price
            
        except Exception as e:
            raise CarPriceException(e, sys) from e

