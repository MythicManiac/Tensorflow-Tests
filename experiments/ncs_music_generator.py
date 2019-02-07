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

from data.ncs_music.dataset import get_dataset


MODEL_ID = 1
NAME = "generator"
SAVE_LOCATION = f"models/ncs_music_generator_{MODEL_ID}.h5"
INPUT_COUNT = 32
OUTPUT_COUNT = 8


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT
        self.n_outputs = OUTPUT_COUNT
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, ), name="in"))

        model.add(Dense(3200, activation="relu"))
        model.add(Dense(1600, activation="relu"))
        model.add(Dense(800, activation="relu"))
        model.add(Dense(400, activation="relu"))
        model.add(Dense(200, activation="relu"))

        model.add(Dense(self.n_outputs, name="out"))

        model.summary()
        model.compile(optimizer="adam", loss="mse")
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
            patience=10,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0.0001
        )
        self.model.fit(
            in_x,
            in_y,
            epochs=100,
            batch_size=4096,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr]
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
        block_interval=8,
        block_size=INPUT_COUNT,
        file_count=40,
        output_size=OUTPUT_COUNT,
    )
    train_data = data.train_data
    test_data = data.test_data

    model = ExperimentalModel()
    model.load()
    # model.train(train_data, data.train_out, test_data, data.test_out)

    for i in range(1):
        result = data.files[i][40]
        inp = result
        result = list(result)

        generate_steps = 4000 * 30
        iterations = int(generate_steps / OUTPUT_COUNT)
        for j in range(iterations):
            if j % 1000 == 0:
                print(f"Progress: {j} / {iterations}")
            out = model.predict_output(inp.reshape(1, *inp.shape))
            result.extend(out[0])
            # inp = out.reshape(32, 1)
            print(j)
            print(inp)
            assert (inp != np.concatenate([inp, out.reshape(OUTPUT_COUNT)])[-INPUT_COUNT:]).all()
            inp = np.concatenate([inp, out.reshape(OUTPUT_COUNT)])[-INPUT_COUNT:]

        data.write_wav(f"output-{NAME}-{MODEL_ID}-{i}.wav", np.array(result))


if __name__ == "__main__":
    main()
