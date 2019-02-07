import os
import wave

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

from data.ncs_music.highfreq_dataset import get_dataset


MODEL_ID = 3
NAME = "autoencoder"
SAVE_LOCATION = f"models/ncs_music_autoencoder_{MODEL_ID}.h5"
INPUT_COUNT = 64
OUTPUT_COUNT = INPUT_COUNT


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT
        self.n_outputs = OUTPUT_COUNT
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))

        encoder = Sequential(name="encoder")
        # encoder.add(Conv1D(10, 3, activation="relu", padding="same"))
        # encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(8, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(16, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(32, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(128, 3, activation="relu", padding="same"))
        model.add(encoder)

        model.add(MaxPool1D(2, name="latent"))

        decoder = Sequential(name="decoder")
        decoder.add(Conv1D(128, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(64, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(32, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(16, 2, activation="relu", padding="same"))  # If this doesn't get loss < 20, remove the smallest 2 layers
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(8, 2, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        # decoder.add(Conv1D(10, 2, activation="relu", padding="same"))
        # decoder.add(UpSampling1D(2))
        model.add(decoder)

        model.add(Conv1D(1, 2, name="out", padding="same"))
        # model.add(Dense(self.n_outputs, name="out"))

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model.load_weights(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, in_x, in_y, val_x, val_y):
        model_checkpoint = ModelCheckpoint(
            SAVE_LOCATION,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0.0001
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=3000,
            batch_size=4096,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr],
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    backend.set_floatx("float16")
    backend.set_epsilon(1e-4)

    data = get_dataset(
        block_interval=1,
        block_size=INPUT_COUNT,
        file_count=1,
        output_size=OUTPUT_COUNT,
        shuffle=True,
    )
    train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    model = ExperimentalModel()
    model.load()
    # model.train(train_data, train_data, test_data, test_data)

    for i in range(1):
        inp = data.files[i].reshape(len(data.files[i]), INPUT_COUNT, 1)
        output = model.predict_output(inp).reshape(len(data.files[i]), INPUT_COUNT)
        data.write_wav(f"output-{NAME}-{MODEL_ID}-{i}.wav", output)
        print(f"output-{NAME}-{MODEL_ID}-{i}.wav created")


if __name__ == "__main__":
    main()
