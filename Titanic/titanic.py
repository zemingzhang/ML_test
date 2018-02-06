import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from pandas import Series,DataFrame

data_train = pd.read_csv("train.csv")

data_train.info()