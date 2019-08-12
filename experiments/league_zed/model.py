import os
import sys
import random

import PySimpleGUI as sg

import numpy as np

from PIL import Image, ImageTk, ImageDraw, ImageFont

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
    add_pool_convolution_2d,
    add_fully_connected,
)


INPUT_SHAPE = (432, 768, 3)
BOTTLENECK_SIZE = 64
OUTPUT_COUNT = 19
TRAINING_DATA_IN_NAME = "data-training-in.npy"
TRAINING_DATA_OUT_NAME = "data-training-out.npy"
VALIDATION_DATA_IN_NAME = "data-validation-in.npy"
VALIDATION_DATA_OUT_NAME = "data-validation-out.npy"
ALL_DATA_IN_NAME = "data-all-in.npy"
ALL_DATA_OUT_NAME = "data-all-out.npy"


def get_all_data():
    return (
        np.load(file=get_input_path(ALL_DATA_IN_NAME), allow_pickle=False),
        np.load(file=get_input_path(ALL_DATA_OUT_NAME), allow_pickle=False),
    )


def get_training_data():
    return (
        (
            np.load(file=get_input_path(TRAINING_DATA_IN_NAME), allow_pickle=False),
            np.load(file=get_input_path(TRAINING_DATA_OUT_NAME), allow_pickle=False),
        ),
        (
            np.load(file=get_input_path(VALIDATION_DATA_IN_NAME), allow_pickle=False),
            np.load(file=get_input_path(VALIDATION_DATA_OUT_NAME), allow_pickle=False),
        )
    )


def get_training_input():
    return (
        np.load(file=get_input_path(TRAINING_DATA_IN_NAME), allow_pickle=False)
    )


def get_validation_input():
    return (
        np.load(file=get_input_path(VALIDATION_DATA_IN_NAME), allow_pickle=False)
    )


class Model(BaseModel):

    def build(self):
        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))

        add_pool_convolution_2d(model, 6)
        add_pool_convolution_2d(model, 12)
        add_pool_convolution_2d(model, 24)
        add_pool_convolution_2d(model, 48)
        model.add(Flatten())
        add_fully_connected(model, BOTTLENECK_SIZE)

        add_fully_connected(model, BOTTLENECK_SIZE * 2)
        add_fully_connected(model, OUTPUT_COUNT)

        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model


def plot(batch_size):
    pass


def train(batch_size, epochs, verbose, data_percentage, patience):
    model = Model()
    print(f"Loading training data")
    training_data, validation_data = get_training_data()

    training_in = training_data[0][:int(len(training_data[0]) * data_percentage)]
    training_out = training_data[1][:int(len(training_data[1]) * data_percentage)]

    validation_in = validation_data[0][:int(len(validation_data[0]) * data_percentage)]
    validation_out = validation_data[1][:int(len(validation_data[1]) * data_percentage)]

    print(f"Beginning training with {batch_size} batch size across {epochs} epochs")
    print(f"Early stop patience: {patience}")
    model.train(
        in_x=training_in,
        in_y=training_out,
        val_x=validation_in,
        val_y=validation_out,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        patience=patience,
    )


def get_action_name(prediction):
    VALID_ACTIONS = (
        "MouseDown.Right",
        "KeyDown.Escape",
        "KeyDown.Space",
        "KeyDown.B",
        "KeyDown.P",
        "KeyDown.D1",
        "KeyDown.D2",
        "KeyDown.D3",
        "KeyDown.D4",
        "KeyDown.D1.Selfcast",
        "KeyDown.D2.Selfcast",
        "KeyDown.D3.Selfcast",
        "KeyDown.D4.Selfcast",
        "KeyDown.Q",
        "KeyDown.W",
        "KeyDown.E",
        "KeyDown.R",
        "KeyDown.Q.Selfcast",
        "KeyDown.W.Selfcast",
        "KeyDown.E.Selfcast",
        "KeyDown.R.Selfcast",
    )
    return VALID_ACTIONS[np.argmax(prediction[2:])]


def draw_cross(image, x, y, size=30):
    draw = ImageDraw.Draw(image)
    half = size / 2
    color = (255, 255, 255, 255)
    draw.line((x - half, y - half, x + half, y + half), width=2, fill=color)
    draw.line((x - half, y + half, x + half, y - half), width=2, fill=color)
    del draw


def add_text(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text((5, 5), text, font=font, fill=(255, 255, 255))


def output(file_count, batch_size):
    if not os.path.exists("output"):
        os.makedirs("output")

    # files = get_validation_input()
    files = get_training_input()
    files = files[:file_count]

    model = Model()
    output = model.predict_output(files, batch_size=batch_size)

    for i, entry in enumerate(output):
        print(f"Outputting prediction... {i + 1}/{len(output)}")
        image = Image.fromarray((files[i] * 255).astype(np.uint8))
        draw_cross(image, entry[0] * image.size[0], entry[1] * image.size[1])
        add_text(image, get_action_name(entry))
        image.save(f"output/{i:05d}.png")


def format_data(file_count):
    basepath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.abspath(os.path.join(basepath, "../data/league_zed"))
    sys.path.append(datapath)
    from dataset import get_dataset
    data = get_dataset(
        file_count=file_count,
        shuffle=True,
    )

    np.save(file=get_output_path(TRAINING_DATA_IN_NAME), arr=data.train_in, allow_pickle=False)
    del data.train_in
    np.save(file=get_output_path(TRAINING_DATA_OUT_NAME), arr=data.train_out, allow_pickle=False)
    del data.train_out

    np.save(file=get_output_path(VALIDATION_DATA_IN_NAME), arr=data.test_in, allow_pickle=False)
    del data.test_in
    np.save(file=get_output_path(VALIDATION_DATA_OUT_NAME), arr=data.test_out, allow_pickle=False)
    del data.test_out

    np.save(file=get_output_path(ALL_DATA_IN_NAME), arr=data.all_in, allow_pickle=False)
    del data.all_in
    np.save(file=get_output_path(ALL_DATA_OUT_NAME), arr=data.all_out, allow_pickle=False)
    del data.all_out


def main():
    args = parse_args()

    if args.format_data:
        format_data(args.file_count)
    else:
        configure_tensorflow()

    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.plot:
        plot(args.batch_size)
    if args.out:
        output(args.file_count, args.batch_size)


if __name__ == "__main__":
    main()
