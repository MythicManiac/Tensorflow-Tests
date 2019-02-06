import os
import wave

import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, InputLayer, Conv1D, MaxPool1D, UpSampling1D
from keras.models import Model, Sequential, load_model
from keras import backend

from data.ncs_music.dataset import get_dataset


SAVE_LOCATION = "models/ncs_music_autoencoder_convolutional.h5"


class AutoEncoder(object):

    def __init__(self):
        self.n_inputs = 2000  # Framerate of 4000 frames per second, 1 second of data
        self.n_features = 500
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))

        encoder = Sequential(name="encoder")
        encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(128, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(512, 3, activation="relu", padding="same"))
        model.add(encoder)

        model.add(MaxPool1D(2, name="latent"))

        decoder = Sequential(name="decoder")
        decoder.add(Conv1D(512, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(128, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(64, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        model.add(decoder)

        model.add(Conv1D(1, 2, name="out", padding="same"))

        model.summary()
        model.compile(optimizer="adam", loss="mse")
        self.model = model

    @property
    def decoder(self):
        if self._decoder:
            return self._decoder

        # https://github.com/keras-team/keras/issues/4811
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(self.n_features,), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        decoder.add(Conv1D(1, 2, name="out", padding="same"))
        decoder.compile(optimizer="adam", loss="mse")
        decoder.summary()

        self._decoder = decoder()

        return decoder

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model = load_model(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, inputs, validation):
        early_stoppping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="auto"
        )
        model_checkpoint = ModelCheckpoint(
            SAVE_LOCATION,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1
        )
        self.model.fit(
            inputs,
            inputs,
            epochs=500,
            batch_size=128,
            validation_data=(validation, validation),
            callbacks=[early_stoppping, model_checkpoint]
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    backend.set_floatx("float16")
    backend.set_epsilon(1e-4)

    data = get_dataset(block_interval=1000, block_size=2000)
    train_data = data.train_data.reshape(len(data.train_data), 2000, 1)
    test_data = data.test_data.reshape(len(data.test_data), 2000, 1)

    model = AutoEncoder()
    model.load()
    model.train(train_data, test_data)
    for i in range(10):
        output = model.predict_output(data.files[i]).reshape(len(data.files[i], 2000))
        data.write_wav(f"output-conv{i}.wav", output)

    # import random
    # inputs = np.array([[random.random() for _ in range(256)] for _ in range(10 * 60)])
    # output = model.decode(inputs)
    # data.write_wav("test-decode.wav", output)


if __name__ == "__main__":
    main()
