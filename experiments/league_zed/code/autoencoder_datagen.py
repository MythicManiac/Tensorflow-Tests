import numpy as np
import keras

from PIL import Image


class DataGenerator(keras.utils.Sequence):

    def __init__(self, filepaths, shape, batch_size):
        self.filepaths = filepaths
        self.shape = shape
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        filepaths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        data = self.load_data(filepaths)
        return data, data

    def load_data(self, filepaths):
        result = np.empty((len(filepaths), *self.shape), dtype=np.float32)
        for i, entry in enumerate(filepaths):
            result[i] = self.load_single_image(entry)
        return result

    def load_single_image(self, path):
        image = Image.open(path)
        return (np.array(image, np.float32, copy=False) / 255)
