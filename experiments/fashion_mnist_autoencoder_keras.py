# Based on https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np

import os
import random

from data.fashion_mnist.dataset import get_dataset


SAVE_LOCATION = "models/fashion_mnist_autoencoder_keras.h5"


class AutoEncoder(object):

    def __init__(self):
        self.build()

    def build(self):
        input_layer = Input(shape=(784,), name="in")

        encoding_layer1 = Dense(1500, activation="relu", name="enc1")(input_layer)
        encoding_layer2 = Dense(1000, activation="relu", name="enc2")(encoding_layer1)
        encoding_layer3 = Dense(500, activation="relu", name="enc3")(encoding_layer2)

        latent_view = Dense(10, activation="sigmoid", name="lat")(encoding_layer3)

        decode_layer1 = Dense(500, activation="relu", name="dec1")(latent_view)
        decode_layer2 = Dense(1000, activation="relu", name="dec2")(decode_layer1)
        decode_layer3 = Dense(1500, activation="relu", name="dec3")(decode_layer2)

        output_layer = Dense(784, name="out")(decode_layer3)

        self.model = Model(input_layer, output_layer)
        self.model.summary()
        self.model.compile(optimizer="adam", loss="mse")

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model = load_model(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, inputs):
        early_stoppping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
        self.model.fit(
            inputs,
            inputs,
            epochs=20,
            batch_size=512,
            validation_data=(inputs, inputs),
            callbacks=[early_stoppping]
        )
        self.model.save(SAVE_LOCATION)

    def plot_data(self, entries):
        f, ax = plt.subplots(1, len(entries))
        f.set_size_inches(10, 5)
        for i in range(len(entries)):
            ax[i].imshow(entries[i].reshape(28, 28))
        plt.show()

    def decode(self, inputs):
        # https://github.com/keras-team/keras/issues/4811
        inp = Input(shape=(10,), name="in")
        dec1 = self.model.get_layer("dec1")
        dec2 = self.model.get_layer("dec2")
        dec3 = self.model.get_layer("dec3")
        out = self.model.get_layer("out")
        decoder = Model(
            inp,
            out(dec3(dec2(dec1(inp))))
        )
        decoder.summary()
        print(inputs.shape)
        return decoder.predict(inputs)

    def predict_output(self, inputs):
        self.plot_data(inputs[:5])
        predictions = self.model.predict(inputs)
        self.plot_data(predictions[:5])


def main():
    data = get_dataset()
    model = AutoEncoder()
    model.load()
    model.train(data.train_data)
    # model.predict_output(data.test_data)
    while True:
        inputs = np.array([[random.random() for _ in range(10)] for _ in range(5)])
        generated = model.decode(inputs)
        model.plot_data(generated)


if __name__ == "__main__":
    main()
