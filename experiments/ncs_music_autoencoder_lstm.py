import os
import wave

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Input,
    InputLayer,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Activation,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Model, Sequential, load_model
from keras import backend

from data.ncs_music.dataset import get_dataset


MODEL_ID = 1
SAVE_LOCATION = f"models/ncs_music_autoencoder_lstm_{MODEL_ID}.h5"
INPUT_COUNT = 32


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT  # Framerate of 4000 frames per second, 1 second of data
        self.n_features = 320
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        encoder = Sequential(name="encoder")
        encoder.add(LSTM(self.n_features, activation="relu", input_shape=(self.n_inputs, 1)))
        model.add(encoder)

        decoder = Sequential(name="decoder")
        decoder.add(RepeatVector(self.n_inputs))
        decoder.add(LSTM(self.n_features, activation="relu", return_sequences=True))
        decoder.add(TimeDistributed(Dense(1)))
        model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse")
        self.model = model

    @property
    def decoder(self):
        if self._decoder:
            return self._decoder

        # https://github.com/keras-team/keras/issues/4811
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(self.n_features, 1), name="in"))
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
            batch_size=4096,
            validation_data=(inputs, inputs),
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

    data = get_dataset(block_interval=1, block_size=INPUT_COUNT, file_count=1)
    train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    model = ExperimentalModel()
    model.load()
    # model.train(train_data, test_data)
    for i in range(min(len(data.files), min(len(data.files), 10))):
        inp = data.files[i].reshape(len(data.files[i]), INPUT_COUNT, 1)
        output = model.predict_output(inp).reshape(len(data.files[i]), INPUT_COUNT)
        data.write_wav(f"output-lstm-{MODEL_ID}-{i}.wav", output)


if __name__ == "__main__":
    main()
