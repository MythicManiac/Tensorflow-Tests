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
SAVE_LOCATION = f"models/ncs_music_lstm_{MODEL_ID}.h5"
INPUT_COUNT = 32
OUTPUT_COUNT = 8


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT  # Framerate of 4000 frames per second, 1 second of data
        self.n_features = 320
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(LSTM(self.n_features, activation="relu", input_shape=(self.n_inputs, 1)))
        model.add(Dense(OUTPUT_COUNT))

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
        self.model.fit(
            in_x,
            in_y,
            epochs=100,
            batch_size=4096,
            validation_data=(val_x, val_y),
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

    data = get_dataset(
        block_interval=8,
        block_size=INPUT_COUNT,
        file_count=40,
        output_size=OUTPUT_COUNT,
    )
    train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    model = ExperimentalModel()
    model.load()
    # model.train(train_data, data.train_out, test_data, data.test_out)

    for i in range(min(len(data.files), min(len(data.files), 10))):
        result = data.files[i][40]
        inp = result.reshape(INPUT_COUNT,  1)
        result = list(result)

        generate_steps = 4000
        iterations = int(generate_steps / OUTPUT_COUNT)
        for j in range(iterations):
            if j % 1000 == 0:
                print(f"Progress: {j} / {iterations}")
            out = model.predict_output(inp.reshape(1, *inp.shape))
            result.extend(out[0])
            # inp = out.reshape(32, 1)
            inp = np.concatenate([inp, out.reshape(OUTPUT_COUNT, 1)])[-INPUT_COUNT:]

        data.write_wav(f"output-lstm-{MODEL_ID}-{i}.wav", np.array(result))


if __name__ == "__main__":
    main()
