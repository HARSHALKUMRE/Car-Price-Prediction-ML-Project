import os, sys
import pandas as pd
import numpy as np
from CarPrice.constant import *
from CarPrice.logger import logging
from CarPrice.exception import CarPriceException
from CarPrice.pipeline.training_pipeline import TrainingPipeline

def main():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")

if __name__ == "__main__":
    main() 