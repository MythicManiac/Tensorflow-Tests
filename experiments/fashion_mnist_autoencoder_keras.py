# Based on https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageTk
import PySimpleGUI as sg

import os
import random

from data.fashion_mnist.dataset import get_dataset


SAVE_LOCATION = "models/fashion_mnist_autoencoder_keras.h5"


class AutoEncoder(object):

    def __init__(self):
        self.build()
        self._decoder = None

    def build(self):
        input_layer = Input(shape=(784,), name="in")

        encoding_layer1 = Dense(1500, activation="relu", name="enc1")(input_layer)
        encoding_layer2 = Dense(1000, activation="relu", name="enc2")(encoding_layer1)
        encoding_layer3 = Dense(500, activation="relu", name="enc3")(encoding_layer2)

        latent_view = Dense(10, activation="sigmoid", name="lat")(encoding_layer3)

        decode_layer1 = Dense(500, activation="relu", name="dec1")(latent_view)
        decode_layer2 = Dense(1000, activation="relu", name="dec2")(decode_layer1)
        decode_layer3 = Dense(1500, activation="relu", name="dec3")(decode_layer2)

        output_layer = Dense(784, name="out")(decode_layer3)

        self.model = Model(input_layer, output_layer)
        self.model.summary()
        self.model.compile(optimizer="adam", loss="mse")

    @property
    def decoder(self):
        if self._decoder:
            return self._decoder

        # https://github.com/keras-team/keras/issues/4811
        inp = Input(shape=(10,), name="in")
        dec1 = self.model.get_layer("dec1")
        dec2 = self.model.get_layer("dec2")
        dec3 = self.model.get_layer("dec3")
        out = self.model.get_layer("out")
        self._decoder = Model(
            inp,
            out(dec3(dec2(dec1(inp))))
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
            batch_size=512,
            validation_data=(inputs, inputs),
            callbacks=[early_stoppping]
        )
        self.model.save(SAVE_LOCATION)

    def plot_data(self, entries):
        f, ax = plt.subplots(1, len(entries))
        f.set_size_inches(10, 5)
        for i in range(len(entries)):
            ax[i].imshow(entries[i].reshape(28, 28))
        plt.show()

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def predict_output(self, inputs):
        self.plot_data(inputs[:5])
        predictions = self.model.predict(inputs)
        self.plot_data(predictions[:5])


class DecoderGUI(object):

    def __init__(self, decoder):
        self.decoder = decoder
        slider_range = (-1000, 1000)
        sliders = [
            [sg.Slider(range=slider_range, default_value=0, size=(12, 8), orientation="horizontal", key=f"input_{i}")]
            for i in range(10)
        ]
        self.layout = [
            [sg.Frame("Values", sliders), sg.Frame("Output", [[sg.Canvas(size=(640, 640), key="canvas")]])],
        ]
        self.window = sg.Window("Fashion MNIST Decoder").Layout(self.layout).Finalize()
        self.canvas = self.window.FindElement("canvas").TKCanvas
        self.image_canvas_id = self.canvas.create_image(640 / 2, 640 / 2)
        self.last_inputs = None
        # self.update_image_from_data(np.linspace(0, 1, 28 * 28) * 255)

    def update_image_from_data(self, data):
        image_data = np.uint8(np.reshape(data, (28, 28)))
        image = Image.fromarray(image_data, "L").resize((640, 640))
        self.image = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_canvas_id, image=self.image)

    def update_picture(self, inputs):
        self.last_inputs = inputs
        output = self.decoder.decode(inputs)
        print(inputs)
        output = output[0]
        output = np.interp(output, (output.min(), output.max()), (0, 255))
        self.update_image_from_data(output)

    def update_values(self, values):
        inputs = []
        for i in range(10):
            key = f"input_{i}"
            inputs.append(values[key] / 1000.0)
        inputs = np.array([inputs])
        if not (inputs == self.last_inputs).all():
            self.update_picture(inputs)

    def event_loop(self):
        while True:
            event, values = self.window.Read(timeout=100)  # Timeout in ms
            if values is None:
                break
            else:
                self.update_values(values)


def main():
    model = AutoEncoder()
    model.load()
    gui = DecoderGUI(model)
    gui.event_loop()
    # data = get_dataset()
    # model = AutoEncoder()
    # model.train(data.train_data)
    # model.predict_output(data.test_data)
    # while True:
    #     inputs = np.array([[random.random() for _ in range(10)] for _ in range(1)])
    #     generated = model.decode(inputs)
    #     model.plot_data(generated)


if __name__ == "__main__":
    main()
