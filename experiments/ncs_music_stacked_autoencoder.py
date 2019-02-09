import os
import wave

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Flatten,
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
    Reshape,
)
from keras.models import Model, Sequential, load_model
from keras import backend

from data.ncs_music.highfreq_dataset import get_dataset, Dataset


MODEL_ID = 1
NAME = "autoencoder_stacked"
INPUT_COUNT = 8
LEVEL_2_INPUT_COUNT = 8
OUTPUT_COUNT = INPUT_COUNT


class Level1Autoencoder(object):

    def __init__(self):
        self.save_location = f"models/ncs_music_autoencoder_stacked_{MODEL_ID}_level1.h5"
        self.n_inputs = INPUT_COUNT
        self.n_outputs = OUTPUT_COUNT
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))

        encoder = Sequential(name="encoder")
        encoder.add(Conv1D(32, 3, padding="same", use_bias=False))
        encoder.add(BatchNormalization())
        encoder.add(Activation("relu"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(64, 3, padding="same", use_bias=False))
        encoder.add(BatchNormalization())
        encoder.add(Activation("relu"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(128, 3, padding="same", use_bias=False))
        encoder.add(BatchNormalization())
        encoder.add(Activation("relu"))
        encoder.add(MaxPool1D(2, name="latent"))
        model.add(encoder)

        decoder = Sequential(name="decoder")
        decoder.add(Conv1D(128, 2, padding="same", use_bias=False))
        decoder.add(BatchNormalization())
        decoder.add(Activation("relu"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(64, 2, padding="same", use_bias=False))
        decoder.add(BatchNormalization())
        decoder.add(Activation("relu"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(32, 2, padding="same", use_bias=False))
        decoder.add(BatchNormalization())
        decoder.add(Activation("relu"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(1, 2, name="out", padding="same", use_bias=False))
        decoder.add(BatchNormalization())
        decoder.add(Activation("relu"))
        model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        if os.path.exists(self.save_location):
            model.load_weights(self.save_location)
            print("Loaded model")

        self.model = model
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))
        encoder.add(self.model.get_layer("encoder"))
        self.encoder = encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(1, 128), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def train(self, in_x, in_y, val_x, val_y):
        model_checkpoint = ModelCheckpoint(
            self.save_location,
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
        early_stoppping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=1,
            mode="auto"
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=40,
            batch_size=4096,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr, early_stoppping],
        )

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


class Level2Autoencoder(object):

    def __init__(self):
        self.save_location = f"models/ncs_music_autoencoder_stacked_{MODEL_ID}_level2.h5"
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(LEVEL_2_INPUT_COUNT, 128), name="in"))

        # encoder = Sequential(name="encoder")
        # encoder.add(Dense(256, activation="relu"))
        # encoder.add(Dense(128, activation="relu"))
        # encoder.add(Dense(64, activation="relu"))
        # encoder.add(Dense(32, activation="sigmoid"))
        # model.add(encoder)

        # decoder = Sequential(name="decoder")
        # decoder.add(Dense(64, activation="relu"))
        # decoder.add(Dense(128, activation="relu"))
        # decoder.add(Dense(256, activation="relu"))
        # decoder.add(Dense(128, activation="relu"))
        # model.add(decoder)

        encoder = Sequential(name="encoder")
        encoder.add(Conv1D(32, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(64, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2))
        encoder.add(Conv1D(128, 3, activation="relu", padding="same"))
        encoder.add(MaxPool1D(2, name="latent"))
        model.add(encoder)

        decoder = Sequential(name="decoder")
        decoder.add(Conv1D(128, 3, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(64, 3, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(32, 3, activation="relu", padding="same"))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(128, 3, name="out", padding="same"))
        model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        if os.path.exists(self.save_location):
            model.load_weights(self.save_location)
            print("Loaded model")

        self.model = model
        self.build_encoder()
        # self.build_decoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(LEVEL_2_INPUT_COUNT, 128), name="in"))
        encoder.add(self.model.get_layer("encoder"))
        self.encoder = encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(LEVEL_2_INPUT_COUNT, 32), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def train(self, in_x, in_y, val_x, val_y):
        model_checkpoint = ModelCheckpoint(
            self.save_location,
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
        early_stoppping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=1,
            mode="auto"
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=10,
            batch_size=128,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr, early_stoppping],
        )

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    backend.set_floatx("float16")
    backend.set_epsilon(1e-4)

    just_files = False
    data = get_dataset(
        block_interval=50,
        block_size=INPUT_COUNT,
        file_count=1,
        output_size=0,
        shuffle=True,
        just_files=just_files,
    )
    if not just_files:
        train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
        test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    level1 = Level1Autoencoder()
    level1.train(train_data, train_data, test_data, test_data)

    # Prepare data by running it through our first level autoencoder
    data = level1.encode(data.files[0].reshape(len(data.files[0]), INPUT_COUNT, 1))

    plotdata = data.reshape(len(data), 128)[:1000]
    plt.subplot(2, 1, 1)
    plt.plot(plotdata)

    data = data[:int(len(data) / LEVEL_2_INPUT_COUNT) * LEVEL_2_INPUT_COUNT]
    data = np.array(np.split(data, len(data) / LEVEL_2_INPUT_COUNT))
    data = data.reshape(len(data), LEVEL_2_INPUT_COUNT, 128)

    # Unload level 1 model
    del level1
    backend.clear_session()

    level2 = Level2Autoencoder()
    level2.train(data, data, data, data)

    output = level2.predict_output(data)
    print(output.shape)

    plotdata = output.reshape(output.shape[0] * output.shape[1], 128)[:1000]
    plt.subplot(2, 1, 2)
    plt.plot(plotdata)
    plt.show()

    print(output.shape)
    output = output.reshape(output.shape[0] * output.shape[1], 1, 128)
    print(output.shape)

    del level2
    backend.clear_session()

    level1 = Level1Autoencoder()
    output = level1.decode(output).flatten()

    data = Dataset()
    data.write_wav(f"output-{NAME}-{MODEL_ID}-level2.wav", output)

    for i in range(min(len(data.files), 2)):
        inp = data.files[i].reshape(len(data.files[i]), INPUT_COUNT, 1)
        output = level1.decode(level1.encode(inp)).reshape(len(data.files[i]), INPUT_COUNT)
        # output = level1.predict_output(inp).reshape(len(data.files[i]), INPUT_COUNT)
        data.write_wav(f"output-{NAME}-{MODEL_ID}-level1-{i}.wav", output)
        print(f"output-{NAME}-{MODEL_ID}-{i}.wav created")
        plt.subplot(2, 2)
        plt.plot(inp.flatten()[2000:8000])
        plt.subplot(2, 2)
        plt.plot(output.flatten()[2000:8000])
        plt.show()


if __name__ == "__main__":
    main()
