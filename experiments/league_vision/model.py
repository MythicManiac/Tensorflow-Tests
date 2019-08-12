import os
import sys
import random

import PySimpleGUI as sg

import numpy as np

from PIL import Image, ImageTk

from keras.layers import (
    InputLayer,
    Flatten,
    Reshape,
)
from keras.models import Sequential

from common import (
    configure_tensorflow,
    parse_args,
    get_input_path,
    get_output_path,
    BaseModel,
    add_convolution_2d,
    add_pool_convolution_2d,
    add_upsampling_convolution_2d,
    add_fully_connected,
)


INPUT_SHAPE = (432, 768, 3)
BOTTLENECK_SIZE = 64
VALIDATION_DATA_NAME = "data-validation.npy"
TRAINING_DATA_NAME = "data-training.npy"
FILES_DATA_NAME = "data-files.npy"

ENCODED_DATA_NAME = "data-encoded.npy"


def get_files_data():
    return np.load(file=get_input_path(FILES_DATA_NAME))


def get_encoded_data():
    return np.load(file=get_input_path(ENCODED_DATA_NAME))


def get_training_data():
    return (
        np.load(file=get_input_path(TRAINING_DATA_NAME), allow_pickle=False),
        np.load(file=get_input_path(VALIDATION_DATA_NAME), allow_pickle=False),
    )


class Model(BaseModel):

    def build(self):
        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))

        encoder = Sequential(name="encoder")
        add_pool_convolution_2d(encoder, 6)
        add_pool_convolution_2d(encoder, 12)
        add_pool_convolution_2d(encoder, 24)
        add_pool_convolution_2d(encoder, 48)
        # add_pool_convolution_2d(encoder, 96)
        encoder.add(Flatten())
        add_fully_connected(encoder, BOTTLENECK_SIZE)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        # add_upsampling_convolution_2d(decoder, 96)
        add_fully_connected(decoder, 27 * 48 * 48)
        decoder.add(Reshape((27, 48, 48)))
        add_upsampling_convolution_2d(decoder, 48)
        add_upsampling_convolution_2d(decoder, 24)
        add_upsampling_convolution_2d(decoder, 12)
        add_upsampling_convolution_2d(decoder, 6)
        add_convolution_2d(decoder, 3)
        model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))
        encoder.add(self.model.get_layer("encoder"))
        self.encoder = encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(BOTTLENECK_SIZE,), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def encode(self, inputs, batch_size):
        return self.encoder.predict(inputs, batch_size)

    def decode(self, inputs, batch_size):
        return self.decoder.predict(inputs, batch_size)


def encode(batch_size):
    files = get_files_data()
    model = Model()
    encoded = model.encode(files, batch_size=batch_size)

    np.save(file=get_output_path(ENCODED_DATA_NAME), arr=encoded, allow_pickle=False)
    del encoded


def plot(batch_size):
    encoded = get_encoded_data()
    pass


def train(batch_size, epochs, verbose, data_percentage, patience):
    model = Model()
    print(f"Loading training data")
    training_data, validation_data = get_training_data()
    training_data = training_data[:int(len(training_data) * data_percentage)]
    validation_data = validation_data[:int(len(validation_data) * data_percentage)]
    print(f"Beginning training with {batch_size} batch size across {epochs} epochs")
    print(f"Early stop patience: {patience}")
    model.train(
        in_x=training_data,
        in_y=training_data,
        val_x=validation_data,
        val_y=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        patience=patience,
    )


def output(file_count, batch_size):
    files_data = get_files_data()
    model = Model()

    print("Predicting output")
    output_data = model.predict_output(files_data, batch_size=batch_size)

    if not os.path.exists("output"):
        os.makedirs("output")

    for i, entry in enumerate(output_data):
        print(f"Outputting image... {i+1}/{len(output_data)}")
        image = Image.fromarray((entry * 255).astype(np.uint8))
        image.save(f"output/{i:04d}.png")


def generate(file_count, batch_size):
    model = Model()

    if not os.path.exists("generated"):
        os.makedirs("generated")

    outputted = 0
    while outputted < file_count:
        values = np.random.uniform(low=0, high=1, size=(batch_size, 512))
        output = model.decode(values, batch_size=batch_size)
        for entry in output:
            image = Image.fromarray((entry * 255).astype(np.uint8))
            image.save(f"generated/{outputted:04d}.png")
            print(f"Generated image... {outputted+1}/{file_count}")
            outputted += 1


def cross(file_count, batch_size):
    if not os.path.exists("crossed"):
        os.makedirs("crossed")

    model = Model()

    inputs = get_training_data()[1]
    encoded = model.encode(inputs, batch_size=batch_size)
    merged = []

    for i in range(file_count):
        a = random.choice(encoded)
        b = random.choice(encoded)
        vals = []
        for i in range(512):
            vals.append(a[i] if random.random() > 0.5 else b[i])
        merged.append(vals)
        # merged.append(a + b / 2)
    merged = np.array(merged)

    output = model.decode(merged, batch_size=batch_size)
    for index, entry in enumerate(output):
        image = Image.fromarray((entry * 255).astype(np.uint8))
        image.save(f"crossed/{index:04d}.png")
        print(f"Crossed image... {index + 1}/{file_count}")


def format_data(file_count):
    basepath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.abspath(os.path.join(basepath, "../data/league"))
    sys.path.append(datapath)
    from dataset import get_dataset
    data = get_dataset(
        file_count=file_count,
        shuffle=True,
    )

    np.save(file=get_output_path(TRAINING_DATA_NAME), arr=data.train_data, allow_pickle=False)
    del data.train_data

    np.save(file=get_output_path(VALIDATION_DATA_NAME), arr=data.test_data, allow_pickle=False)
    del data.test_data

    np.save(file=get_output_path(FILES_DATA_NAME), arr=data.files, allow_pickle=False)


def gui():
    model = Model()
    gui = DecoderGUI(model)
    gui.event_loop()


def main():
    args = parse_args()

    if args.format_data:
        format_data(args.file_count)
    else:
        configure_tensorflow()

    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.encode:
        encode(args.batch_size)
    if args.generate:
        generate(args.file_count, args.batch_size)
    if args.cross:
        cross(args.file_count, args.batch_size)
    if args.plot:
        plot(args.batch_size)
    if args.gui:
        gui()
    if args.out:
        output(args.file_count, args.batch_size)


class DecoderGUI(object):

    def __init__(self, decoder):
        self.decoder = decoder
        slider_range = (-1000, 1000)
        sliders = [
            [sg.Slider(range=slider_range, default_value=0, size=(12, 8), orientation="horizontal", key=f"input_{i}")]
            for i in range(BOTTLENECK_SIZE)
        ]
        self.layout = [
            [sg.Frame("Values", sliders), sg.Frame("Output", [[sg.Canvas(size=(640, 640), key="canvas")]])],
        ]
        self.window = sg.Window("League Decoder").Layout(self.layout).Finalize()
        self.canvas = self.window.FindElement("canvas").TKCanvas
        self.image_canvas_id = self.canvas.create_image(640 / 2, 640 / 2)
        self.last_inputs = None

    def update_image_from_data(self, data):
        data = (data * 255).astype(np.uint8)
        image = Image.fromarray(data)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_canvas_id, image=self.image)

    def update_picture(self, inputs):
        self.last_inputs = inputs
        output = self.decoder.decode(inputs, batch_size=1)
        # output = output[0]
        # output = np.interp(output, (output.min(), output.max()), (0, 255))
        self.update_image_from_data(output[0])

    def update_values(self, values):
        inputs = []
        for i in range(BOTTLENECK_SIZE):
            key = f"input_{i}"
            inputs.append(values[key] / 100.0)
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


if __name__ == "__main__":
    main()
