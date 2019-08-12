import os
import argparse

import numpy as np


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.layers import (
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    PReLU,
    Dense,
)

from lr_scheduler import SGDRScheduler


def add_fully_connected(sequential, neurons):
    sequential.add(Dense(neurons))
    sequential.add(BatchNormalization())
    sequential.add(PReLU())


def add_convolution(sequential, filters):
    sequential.add(Conv1D(filters, 3, padding="same", use_bias=False))
    sequential.add(BatchNormalization())
    sequential.add(PReLU())


def add_pool_convolution(sequential, filters):
    add_convolution(sequential, filters)
    sequential.add(MaxPool1D(2))


def add_upsampling_convolution(sequential, filters):
    add_convolution(sequential, filters)
    sequential.add(UpSampling1D(2))


def add_convolution_2d(sequential, filters):
    sequential.add(Conv2D(filters, (3, 3), padding="same", use_bias=False))
    sequential.add(BatchNormalization())
    sequential.add(PReLU())


def add_pool_convolution_2d(sequential, filters):
    add_convolution_2d(sequential, filters)
    sequential.add(MaxPooling2D(pool_size=(2, 2)))


def add_upsampling_convolution_2d(sequential, filters):
    add_convolution_2d(sequential, filters)
    sequential.add(UpSampling2D(size=(2, 2)))


def get_input_path(name):
    valohai_input_name = name.split(".")[0]
    valohai_path = os.path.abspath(os.path.join("/valohai/inputs/", valohai_input_name, name))
    if os.path.isfile(valohai_path):
        print(f"Using input {valohai_path}")
        return os.path.abspath(valohai_path)
    else:
        print(f"No input found at {valohai_path}, using local")
    basepath = os.path.dirname(os.path.abspath(__file__))
    result = os.path.abspath(os.path.join(basepath, name))
    return result


def get_output_path(name):
    result = ""
    valohai_output = "/valohai/outputs/"
    if os.path.isdir(valohai_output):
        valohai_path = os.path.join("/valohai/outputs/", name)
        result = os.path.abspath(valohai_path)
    else:
        basepath = os.path.dirname(os.path.abspath(__file__))
        result = os.path.abspath(os.path.join(basepath, name))
    print(f"Using output path {result}")
    return result


def configure_tensorflow():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format-data", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--data-percentage", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--out", action="store_true")
    parser.add_argument("--file-count", type=int, default=1)
    return parser.parse_args()


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


class BaseModel(object):

    def __init__(self, summary=True):
        name = self.__class__.__name__
        self.weights_load_location = get_input_path(f"weights-{name}.h5")
        self.weights_save_location = get_output_path(f"weights-{name}.h5")
        self.build()
        self.load()
        if summary:
            self.model.summary()

    def load(self):
        if os.path.exists(self.weights_load_location):
            self.model.load_weights(self.weights_load_location)
            print(f"Loaded a model from {self.weights_load_location}")

    def train(self, in_x, in_y, val_x, val_y, epochs, batch_size, verbose, patience):
        model_checkpoint = ModelCheckpoint(
            self.weights_save_location,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1
        )
        early_stoppping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=1,
            mode="auto"
        )
        schedule = SGDRScheduler(
            min_lr=1e-5,
            max_lr=1e-2,
            steps_per_epoch=np.ceil(epochs / batch_size),
            lr_decay=0.9,
            cycle_length=5,
            mult_factor=1.5
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, schedule, early_stoppping],
            verbose=verbose,
        )

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs, batch_size=10):
        return self.model.predict(inputs, batch_size=batch_size)
