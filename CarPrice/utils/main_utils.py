import os
import sys
import shutil
from typing import Dict, Tuple

import yaml
from yaml import safe_dump
from CarPrice.exception import CarPriceException
from CarPrice.logger import logging
from pandas import DataFrame

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CarPriceException(e, sys) from e