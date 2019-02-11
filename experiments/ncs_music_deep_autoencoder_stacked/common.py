import os
import argparse

import numpy as np

import tensorflow as tf

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.backend.tensorflow_backend import set_session


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

    def __init__(self):
        name = self.__class__.__name__
        self.weights_load_location = get_input_path(f"weights-{name}.h5")
        self.weights_save_location = get_output_path(f"weights-{name}.h5")
        self.build()
        self.load()

    def load(self):
        if os.path.exists(self.weights_load_location):
            self.model.load_weights(self.weights_load_location)
            print("Loaded a model")

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
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=patience,
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
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr, early_stoppping],
            verbose=verbose,
        )

    def predict_output(self, inputs):
        return self.model.predict(inputs)
