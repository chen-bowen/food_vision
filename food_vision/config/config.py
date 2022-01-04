import pathlib

import food_vision
import pandas as pd

pd.options.display.max_rows = 100
pd.options.display.max_columns = 50

PACKAGE_ROOT = pathlib.Path(food_vision.__file__).resolve().parent


# MODEL FITTING
IMAGE_SIZE = 224
NUM_CLASSES = 101
BATCH_SIZE = 32
NUM_EPOCHS = 3  # 1 for testing, 10 for final model
LEARNING_RATE = 0.001

# MODEL PERSISTING
MODEL_PATH = "../trained_models/trained_model"
FINE_TUNED_MODEL_PATH = "../trained_models/fine_tuned_model"
