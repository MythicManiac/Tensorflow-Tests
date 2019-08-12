import os
import random

import numpy as np

from PIL import Image


class Dataset(object):

    def get_path(self, path):
        basepath = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(basepath, path)
        return csv_path

    def get_all_files(self, path):
        path = self.get_path(path)
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        return files

    def read_bmp(self, path):
        image = Image.open(path)
        return (np.array(image).astype(np.float32) / 255)

    def write_bmp(self, path, data):
        image = Image.fromarray((data * 255).astype(np.uint8))
        image.save(path)

    def __init__(self, file_count=None, shuffle=True):
        files = self.get_all_files("league_frames")
        # if shuffle:
        #     random.shuffle(files)

        if file_count is None:
            file_count = len(files)
        file_count = min(len(files), file_count)

        data = []
        for i in range(file_count):
            print(f"Processing file {i + 1} / {file_count}")
            data.append(self.read_bmp(files[i]))

        self.files = np.array(data)

        if shuffle:
            np.random.shuffle(data)

        cutoff = int(len(data) * 0.8)
        self.train_data = np.array(data[:cutoff])
        # self.train_out = np.array(outputs[:cutoff])
        self.test_data = np.array(data[cutoff:])
        # self.test_out = np.array(outputs[cutoff:])
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")


def get_dataset(file_count=None, shuffle=True):
    return Dataset(file_count=file_count, shuffle=shuffle)
