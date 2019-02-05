import os
import wave

import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras import backend

from matplotlib import pyplot

from data.ncs_music.dataset import get_dataset


SAVE_LOCATION = "models/ncs_music_autoencoder.h5"


class AutoEncoder(object):

    def __init__(self):
        self.n_inputs = 8000
        self.n_features = 512
        self._decoder = None
        self.build()

    def build(self):
        input_layer = Input(shape=(self.n_inputs,), name="in")

        encoding_layer1 = Dense(6000, activation="relu", name="enc1")(input_layer)
        encoding_layer2 = Dense(4000, activation="relu", name="enc2")(encoding_layer1)
        encoding_layer3 = Dense(2000, activation="relu", name="enc3")(encoding_layer2)
        encoding_layer4 = Dense(1000, activation="relu", name="enc4")(encoding_layer3)
        encoding_layer5 = Dense(500, activation="relu", name="enc5")(encoding_layer4)

        latent_view = Dense(self.n_features, activation="sigmoid", name="lat")(encoding_layer5)

        decode_layer1 = Dense(500, activation="relu", name="dec1")(latent_view)
        decode_layer2 = Dense(1000, activation="relu", name="dec2")(decode_layer1)
        decode_layer3 = Dense(2000, activation="relu", name="dec3")(decode_layer2)
        decode_layer4 = Dense(4000, activation="relu", name="dec4")(decode_layer3)
        decode_layer5 = Dense(6000, activation="relu", name="dec5")(decode_layer4)

        output_layer = Dense(self.n_inputs, name="out")(decode_layer5)

        self.model = Model(input_layer, output_layer)
        self.model.summary()
        self.model.compile(optimizer="adam", loss="mse")

    @property
    def decoder(self):
        if self._decoder:
            return self._decoder

        # https://github.com/keras-team/keras/issues/4811
        inp = Input(shape=(self.n_inputs,), name="in")
        dec1 = self.model.get_layer("dec1")
        dec2 = self.model.get_layer("dec2")
        dec3 = self.model.get_layer("dec3")
        dec4 = self.model.get_layer("dec4")
        dec5 = self.model.get_layer("dec5")
        out = self.model.get_layer("out")
        self._decoder = Model(
            inp,
            out(dec5(dec4(dec3(dec2(dec1(inp))))))
        )
        self._decoder.summary()
        return self._decoder

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model = load_model(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, inputs):
        early_stoppping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
        self.model.fit(
            inputs,
            inputs,
            epochs=50,
            batch_size=5,
            validation_data=(inputs, inputs),
            callbacks=[early_stoppping]
        )
        self.model.save(SAVE_LOCATION)

    def plot_data(self, entries):
        f, ax = pyplot.subplots(1, len(entries))
        f.set_size_inches(10, 5)
        for i in range(len(entries)):
            ax[i].imshow(entries[i].reshape(28, 28))
        pyplot.show()

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        # self.plot_data(inputs[:5])
        predictions = self.model.predict(inputs)
        return predictions
        # self.plot_data(predictions[:5])


def write_output(output_data, params, target):
    with wave.open("output.wav", "wb") as f:
        f.setparams(params)
        f.writeframes(bytes((output_data.flatten() * 255).astype(np.byte)))


def main():
    backend.set_floatx("float16")
    backend.set_epsilon(1e-4)
    data = get_dataset()
    model = AutoEncoder()
    model.load()
    # model.train(data.train_data)
    output = model.predict_output(data.first_song)
    write_output(output, data.first_params, "output.wav")

    # model.predict_output(data.train_data)
    # while True:
    #     inputs = np.array([[random.random() for _ in range(10)] for _ in range(1)])
    #     generated = model.decode(inputs)
    #     model.plot_data(generated)


if __name__ == "__main__":
    main()
