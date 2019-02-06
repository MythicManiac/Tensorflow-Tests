import os

import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras import backend

from data.ncs_music.dataset import get_dataset


SAVE_LOCATION = "models/ncs_music_autoencoder.h5"


class AutoEncoder(object):

    def __init__(self):
        self.n_inputs = 800
        self.n_features = 1600
        self._decoder = None
        self.build()

    def build(self):
        input_layer = Input(shape=(self.n_inputs,), name="in")

        encoding_layer1 = Dense(1800, activation="relu", name="enc1")(input_layer)
        encoding_layer2 = Dense(1200, activation="relu", name="enc2")(encoding_layer1)
        encoding_layer3 = Dense(800, activation="relu", name="enc3")(encoding_layer2)
        encoding_layer4 = Dense(400, activation="relu", name="enc4")(encoding_layer3)

        latent_view = Dense(self.n_features, activation="sigmoid", name="lat")(encoding_layer4)

        decode_layer1 = Dense(400, activation="relu", name="dec1")(latent_view)
        decode_layer2 = Dense(800, activation="relu", name="dec2")(decode_layer1)
        decode_layer3 = Dense(1200, activation="relu", name="dec3")(decode_layer2)
        decode_layer4 = Dense(1800, activation="relu", name="dec4")(decode_layer3)

        output_layer = Dense(self.n_inputs, name="out")(decode_layer4)

        self.model = Model(input_layer, output_layer)
        self.model.summary()
        self.model.compile(optimizer="adam", loss="mse")

    @property
    def decoder(self):
        if self._decoder:
            return self._decoder

        # https://github.com/keras-team/keras/issues/4811
        inp = Input(shape=(self.n_features,), name="in")
        dec1 = self.model.get_layer("dec1")
        dec2 = self.model.get_layer("dec2")
        dec3 = self.model.get_layer("dec3")
        dec4 = self.model.get_layer("dec4")
        out = self.model.get_layer("out")
        self._decoder = Model(
            inp,
            out(dec4(dec3(dec2(dec1(inp)))))
        )
        self._decoder.summary()
        return self._decoder

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model = load_model(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, inputs, validation):
        early_stoppping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="auto"
        )
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
            epochs=200,
            batch_size=128,
            validation_data=(validation, validation),
            callbacks=[model_checkpoint]
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions


def main():
    data = get_dataset(block_interval=200, block_size=800, file_count=107)
    model = AutoEncoder()
    model.load()
    # model.train(data.train_data, data.test_data)
    for i in range(10):
        output = model.predict_output(data.files[i])
        data.write_wav(f"output{i}.wav", output)

    # import random
    # inputs = np.array([[random.random() for _ in range(256)] for _ in range(10 * 60)])
    # output = model.decode(inputs)
    # data.write_wav("test-decode.wav", output)


if __name__ == "__main__":
    main()
