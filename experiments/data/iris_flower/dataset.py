import os

import numpy as np
import pandas as pd


class Dataset(object):

    def __init__(self, data, test_share):
        test_data = data[:test_share]
        train_data = data[test_share:]
        self.labels = ["Setosa", "Versicolor", "Virginica"]

        self.test_data_in = test_data.drop("Species", axis=1)
        self.test_data_out = test_data["Species"]
        self.test_data_labels = (np.arange(3) == self.test_data_out[:, None]).astype(np.float32)

        self.train_data_in = train_data.drop("Species", axis=1)
        self.train_data_out = train_data["Species"]
        self.train_data_labels = (np.arange(3) == self.train_data_out[:, None]).astype(np.float32)


def get_csv_path():
    basepath = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(basepath, "iris_flowers.csv")
    return csv_path


def get_dataset():
    return Dataset(pd.read_csv(get_csv_path()), 30)
