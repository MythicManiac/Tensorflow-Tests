import os
import sys

import numpy as np

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    InputLayer,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Activation,
)
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session


def get_input_path(name):
    valohai_path = os.path.join("/valohai/inputs/", name)
    if os.path.isfile(valohai_path):
        return os.path.abspath(valohai_path)
    basepath = os.path.dirname(os.path.abspath(__file__))
    result = os.path.abspath(os.path.join(basepath, name))
    return result


def get_output_path(name):
    valohai_output = "/valohai/outputs/"
    if os.path.isdir(valohai_output):
        valohai_path = os.path.join("/valohai/outputs/", name)
        return os.path.abspath(valohai_path)
    basepath = os.path.dirname(os.path.abspath(__file__))
    result = os.path.abspath(os.path.join(basepath, name))
    return result


VALIDATION_DATA_PATH = get_input_path("data-validation.npy")
TRAINING_DATA_PATH = get_input_path("data-training.npy")
FILES_DATA_PATH = get_input_path("data-files.npy")
WEIGHTS_LOAD_PATH = get_input_path("weights.h5")
WEIGHTS_SAVE_PATH = get_output_path("weights.h5")
INPUT_COUNT = 729


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))

        def add_convolution(sequential, filters):
            sequential.add(Conv1D(filters, 3, padding="same", use_bias=False))
            sequential.add(BatchNormalization())
            sequential.add(Activation("relu"))

        encoder = Sequential(name="encoder")
        add_convolution(encoder, 32)
        encoder.add(MaxPool1D(3))
        add_convolution(encoder, 64)
        encoder.add(MaxPool1D(3))
        add_convolution(encoder, 64)
        encoder.add(MaxPool1D(3))
        add_convolution(encoder, 128)
        encoder.add(MaxPool1D(3))
        add_convolution(encoder, 128)
        encoder.add(MaxPool1D(3))
        add_convolution(encoder, 256)
        encoder.add(MaxPool1D(3))
        model.add(encoder)

        decoder = Sequential(name="decoder")
        add_convolution(decoder, 256)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 128)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 128)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 64)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 64)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 32)
        decoder.add(UpSampling1D(3))
        add_convolution(decoder, 1)
        model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model
        self.load()

    def load(self):
        if os.path.exists(WEIGHTS_LOAD_PATH):
            self.model.load_weights(WEIGHTS_LOAD_PATH)
            print("Loaded a model")

    def train(self, in_x, in_y, val_x, val_y):
        model_checkpoint = ModelCheckpoint(
            WEIGHTS_LOAD_PATH,
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
            patience=100,
            verbose=1,
            mode="auto"
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=10,
            batch_size=2048,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr, early_stoppping],
        )

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def get_files_data():
    return np.load(file=FILES_DATA_PATH)


def get_training_data():
    return (
        np.load(file=TRAINING_DATA_PATH, allow_pickle=False),
        np.load(file=VALIDATION_DATA_PATH, allow_pickle=False),
    )


def plot():
    import matplotlib.pyplot as plt
    files = get_files_data()
    model = ExperimentalModel()
    plt.subplot(2, 2, 1)
    plt.plot(files[0][200])

    plt.subplot(2, 2, 2)
    plt.plot(model.predict_output(files[0][200].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.subplot(2, 2, 3)
    plt.plot(files[0][210])

    plt.subplot(2, 2, 4)
    plt.plot(model.predict_output(files[0][210].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.show()


def train():
    training_data, validation_data = get_training_data()
    model = ExperimentalModel()
    model.train(training_data, training_data, validation_data, validation_data)


def output():
    files = get_files_data()
    model = ExperimentalModel()
    for i in range(min(len(files), 10)):
        inp = files[i].reshape(len(files[i]), INPUT_COUNT, 1)
        output = model.predict_output(inp).flatten()
        write_wav(f"output-{i}.wav", output)
        print(f"output-{i}.wav created")


def format_training_data():
    basepath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.abspath(os.path.join(basepath, "../data/ncs_music"))
    sys.path.append(datapath)
    from highfreq_dataset import get_dataset
    data = get_dataset(
        block_interval=int(INPUT_COUNT / 2),
        block_size=INPUT_COUNT,
        file_count=55,
        output_size=0,
        shuffle=True,
        just_files=False,
    )
    data.train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1).astype(np.float16)
    data.test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1).astype(np.float16)
    np.save(file=TRAINING_DATA_PATH, arr=data.train_data, allow_pickle=False)
    np.save(file=VALIDATION_DATA_PATH, arr=data.test_data, allow_pickle=False)
    np.save(file=FILES_DATA_PATH, arr=data.files)


def write_wav(path, data):
    import wave
    params = (
        1,                 # Channel count
        1,                 # Sample width
        44100,             # Frame rate
        len(data),         # Frame count
        "NONE",            # Compression type
        "not compressed",  # Compression name
    )

    # Convert back to bytes
    data = bytes((data * 255).astype(np.byte))

    with wave.open(path, "wb") as f:
        f.setparams(params)
        f.writeframes(data)


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

    if "--format-data" in sys.argv:
        format_training_data()
    if "--train" in sys.argv:
        train()
    if "--plot" in sys.argv:
        plot()
    if "--out" in sys.argv:
        output()


if __name__ == "__main__":
    main()
