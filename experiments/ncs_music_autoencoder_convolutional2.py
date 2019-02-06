import os
import wave

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, InputLayer, Conv1D, MaxPool1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, load_model
from keras import backend

from data.ncs_music.dataset import get_dataset


CONV_ID = 2
SAVE_LOCATION = f"models/ncs_music_autoencoder_convolutional_{CONV_ID}.h5"
INPUT_COUNT = 400


class AutoEncoder(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT  # Framerate of 4000 frames per second, 1 second of data
        self.n_features = 100
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, ), name="in"))

        encoder = Sequential(name="encoder")
        encoder.add(Dense(800, activation="relu"))
        encoder.add(BatchNormalization())
        encoder.add(Dense(400, activation="relu"))
        encoder.add(BatchNormalization())
        encoder.add(Dense(200, activation="relu"))
        encoder.add(BatchNormalization())
        model.add(encoder)

        model.add(Dense(self.n_features, activation="sigmoid", name="latent"))

        decoder = Sequential(name="decoder")
        decoder.add(Dense(200, activation="relu"))
        decoder.add(BatchNormalization())
        decoder.add(Dense(400, activation="relu"))
        decoder.add(BatchNormalization())
        decoder.add(Dense(800, activation="relu"))
        decoder.add(BatchNormalization())
        model.add(decoder)

        model.add(Dense(self.n_inputs, name="out"))

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
            self.model.load_weights(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, inputs, validation):
        # early_stoppping = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0,
        #     patience=10,
        #     verbose=1,
        #     mode="auto"
        # )
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
            epochs=100,
            batch_size=512,
            validation_data=(inputs, inputs),
            # callbacks=[early_stoppping, model_checkpoint]
            callbacks=[model_checkpoint]
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    backend.set_floatx("float16")
    backend.set_epsilon(1e-4)

    data = get_dataset(block_interval=int(INPUT_COUNT / 4), block_size=INPUT_COUNT, file_count=10)
    # train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    # test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)
    train_data = data.train_data
    test_data = data.test_data

    # plt.plot(train_data.flatten())
    # plt.show()

    # plot_data = np.bincount((train_data.flatten() * 255).astype(int))

    # plt.plot(plot_data)
    # plt.show()

    # print(plot_data.shape)

    model = AutoEncoder()
    model.load()
    model.train(train_data, test_data)
    for i in range(min(len(data.files), 10)):
        output = model.predict_output(data.files[i])
        data.write_wav(f"output-conv-{CONV_ID}-{i}.wav", output)

    # import random
    # inputs = np.array([[random.random() for _ in range(256)] for _ in range(10 * 60)])
    # output = model.decode(inputs)
    # data.write_wav("test-decode.wav", output)


if __name__ == "__main__":
    main()
