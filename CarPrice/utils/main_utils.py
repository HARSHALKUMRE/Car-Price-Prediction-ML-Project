import os
import sys
import shutil
from typing import Dict, Tuple

import yaml
import dill
from yaml import safe_dump
from CarPrice.constant import SCHEMA_FILE_PATH
from CarPrice.exception import CarPriceException
from CarPrice.logger import logging
from pandas import DataFrame
import pandas as pd
import numpy as np

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CarPriceException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CarPriceException(e, sys) from e

def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of MainUtils class")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of MainUtils class")

        return obj

    except Exception as e:
        raise CarPriceException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CarPriceException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CarPriceException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of MainUtils class")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CarPriceException(e, sys) from e

def get_carlist():
    try:
        data_url = "https://raw.githubusercontent.com/HARSHALKUMRE/Main-Branching/main/cardekho_dataset.csv"
        df = pd.read_csv(data_url)
        car_list = list(df.car_name.unique())
        return car_list
    except Exception as e:
        raise CarPriceException(e, sys) from e