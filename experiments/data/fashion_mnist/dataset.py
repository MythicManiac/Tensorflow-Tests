# Dataset is from https://www.kaggle.com/zalando-research/fashionmnist/

import os

import numpy as np


class Dataset(object):

    @property
    def train_data_path(self):
        return self.get_path("fashion-mnist_train.csv")

    @property
    def test_data_path(self):
        return self.get_path("fashion-mnist_test.csv")

    def get_path(self, filename):
        basepath = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(basepath, filename)
        return csv_path

    def __init__(self):
        self.train_data = np.loadtxt(
            self.train_data_path,
            delimiter=",",
            skiprows=1
        )[:, 1:]

        self.test_data = np.loadtxt(
            self.test_data_path,
            delimiter=",",
            skiprows=1
        )[:, 1:]


def get_dataset():
    return Dataset()
