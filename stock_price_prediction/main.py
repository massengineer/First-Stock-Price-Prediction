import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

data = pd.read_csv(
    r"C:\Users\louis\Documents\PythonPortfolio\FirstPythonProj\data\raw\all_stocks_5yr.csv\all_stocks_5yr.csv",
    delimiter=",",
    on_bad_lines="skip",
)
print(data.shape)
print(data.sample(7))
