import os
import sys

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
NAME = "autoencoder_deep"
SAVE_LOCATION = f"models/ncs_music_autoencoder_deep_{MODEL_ID}.h5"
# INPUT_COUNT = 16384
# INPUT_COUNT = 6561
INPUT_COUNT = 729
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

        # encoder = Sequential(name="encoder")
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3))
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3))
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3))
        model.add(Conv1D(256, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(3, name="encoder_out"))
        # model.add(encoder)

        # model.add(Conv1D(2048, 3, padding="same", use_bias=False))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))
        # model.add(MaxPool1D(256))
        # model.add(Conv1D(2048, 3, padding="same", use_bias=False))
        # model.add(BatchNormalization())
        # model.add(Activation("relu"))
        # model.add(UpSampling1D(256))

        # decoder = Sequential(name="decoder")
        model.add(Conv1D(256, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(3))
        model.add(Conv1D(1, 3, name="decoder_out", padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(decoder)

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
        # early_stoppping = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0,
        #     patience=5,
        #     verbose=1,
        #     mode="auto"
        # )
        self.model.fit(
            in_x,
            in_y,
            epochs=3000,
            batch_size=128,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr],
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    # backend.set_floatx("float16")
    # backend.set_epsilon(1e-4)

    data = get_dataset(
        block_interval=max(min(INPUT_COUNT, 10000), 16),
        block_size=INPUT_COUNT,
        file_count=30,
        output_size=0,
        shuffle=True,
    )
    train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    model = ExperimentalModel()
    model.load()
    if "--train" in sys.argv:
        model.train(train_data, train_data, test_data, test_data)

    plt.subplot(2, 2, 1)
    plt.plot(data.files[0][200])

    plt.subplot(2, 2, 2)
    plt.plot(model.predict_output(data.files[0][200].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.subplot(2, 2, 3)
    plt.plot(data.files[0][210])

    plt.subplot(2, 2, 4)
    plt.plot(model.predict_output(data.files[0][210].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.show()

    for i in range(min(len(data.files), 10)):
        inp = data.files[i].reshape(len(data.files[i]), INPUT_COUNT, 1)
        output = model.predict_output(inp).flatten()
        data.write_wav(f"output-{NAME}-{MODEL_ID}-{i}.wav", output)
        print(f"output-{NAME}-{MODEL_ID}-{i}.wav created")


if __name__ == "__main__":
    main()
