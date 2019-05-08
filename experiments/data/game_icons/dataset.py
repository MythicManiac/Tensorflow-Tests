import os

from PIL import Image

import numpy as np


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

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path

    def process_data(self):
        files = self.get_all_files(self.input_path)
        for file in files:
            output_path = os.path.abspath(os.path.join(
                self.output_path,
                os.path.split(file)[1]
            ))
            print(output_path)
            image = Image.open(file)
            image.thumbnail((128, 128))
            image.save(output_path)

    def load(self):
        files = self.get_all_files(self.input_path)
        data = []
        for file in files:
            data.append(np.array(Image.open(file).convert("RGB")) / 255)
        data = np.array(data)
        np.random.shuffle(data)

        cutoff = int(len(data) * 0.8)
        self.training_data = np.array(data[:cutoff])
        self.validation_data = np.array(data[cutoff:])


def get_dataset():
    dataset = Dataset(input_path="icons")
    dataset.load()
    return dataset


def process_data():
    dataset = Dataset(input_path="icons_unprocessed", output_path="icons")
    dataset.process_data()
    return dataset
