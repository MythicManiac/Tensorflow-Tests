import os
import sys

import numpy as np

from PIL import Image

from keras.layers import (
    InputLayer,
    # Flatten,
    # Reshape,
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
    # add_fully_connected,
)


INPUT_SHAPE = (432, 768, 3)
BOTTLENECK_SIZE = 128
VALIDATION_DATA_NAME = "data-validation.npy"
TRAINING_DATA_NAME = "data-training.npy"
FILES_DATA_NAME = "data-files.npy"


def get_files_data():
    return np.load(file=get_input_path(FILES_DATA_NAME))


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
        add_pool_convolution_2d(encoder, 32)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        add_upsampling_convolution_2d(decoder, 32)
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
        decoder.add(InputLayer(input_shape=(216, 384, 32,), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)


def plot():
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


def main():
    args = parse_args()

    if args.format_data:
        format_data(args.file_count)
    else:
        configure_tensorflow()

    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.plot:
        plot()
    if args.out:
        output(args.file_count, args.batch_size)


if __name__ == "__main__":
    main()
