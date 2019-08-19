import os

import numpy as np

from PIL import Image

from keras.layers import (
    InputLayer,
    Flatten,
    Reshape,
)
from keras.models import Sequential

from .common import (
    configure_tensorflow,
    parse_args,
    BaseModel,
    add_convolution_2d,
    add_pool_convolution_2d,
    add_upsampling_convolution_2d,
    add_fully_connected,
)

from .autoencoder_datagen import DataGenerator


INPUT_SHAPE = (288, 512, 3)
BOTTLENECK_SIZE = 256
# BOTTLENECK_SIZE = 64


class Autoencoder(BaseModel):

    def build(self):
        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))

        encoder = Sequential(name="encoder")
        add_pool_convolution_2d(encoder, 6)
        add_pool_convolution_2d(encoder, 12)
        add_pool_convolution_2d(encoder, 24)
        add_pool_convolution_2d(encoder, 48)
        add_pool_convolution_2d(encoder, 96)
        # encoder.add(Flatten())
        # add_fully_connected(encoder, BOTTLENECK_SIZE)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        # add_fully_connected(decoder, 9 * 16 * 96)
        # decoder.add(Reshape((9, 16, 96)))
        add_upsampling_convolution_2d(decoder, 96)
        add_upsampling_convolution_2d(decoder, 48)
        add_upsampling_convolution_2d(decoder, 24)
        add_upsampling_convolution_2d(decoder, 12)
        add_upsampling_convolution_2d(decoder, 6)
        add_convolution_2d(decoder, 3)
        model.add(decoder)

        encoder.summary()
        decoder.summary()

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
        decoder.add(InputLayer(input_shape=(9, 16, 96), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def encode(self, inputs, batch_size):
        return self.encoder.predict(inputs, batch_size)

    def decode(self, inputs, batch_size):
        return self.decoder.predict(inputs, batch_size)


def get_path(path):
    basepath = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(basepath, path)
    return csv_path


def get_all_files(path, ext=None):
    path = get_path(path)
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if (os.path.isfile(os.path.join(path, f)) and (ext is None or f.endswith(ext)))
    ]
    for entry in os.listdir(path):
        next_path = os.path.join(path, entry)
        if os.path.isdir(next_path):
            files += get_all_files(next_path, ext=ext)
    return files


def get_generators(batch_size):
    filepaths = get_all_files("../data", ext=".jpg")
    cutoff = int(len(filepaths) * 0.8)
    training_data = DataGenerator(
        batch_size=batch_size,
        filepaths=filepaths[:cutoff],
        shape=INPUT_SHAPE,
    )
    validation_data = DataGenerator(
        batch_size=batch_size,
        filepaths=filepaths[cutoff:],
        shape=INPUT_SHAPE,
    )
    return (training_data, validation_data)


def train(batch_size, epochs, verbose, patience):
    print(f"Loading training data")
    training_data, validation_data = get_generators(batch_size)
    model = Autoencoder()
    print(f"Beginning training with {batch_size} batch size across {epochs} epochs")
    print(f"Early stop patience: {patience}")

    model.fit_generator(
        generator=training_data,
        validation_data=validation_data,
        use_multiprocessing=False,
        epochs=epochs,
        verbose=verbose,
        patience=patience,
        batch_size=batch_size,
        max_queue_size=1,
    )


def output(file_count, batch_size):
    if not os.path.exists("output"):
        os.makedirs("output")

    training_data, validation_data = get_generators(batch_size)
    model = Autoencoder()

    outputted = 0
    for batch_in, batch_out in validation_data:
        output = model.predict_output(batch_in, batch_size=batch_size)
        for entry in output:
            print(f"Outputting prediction... {outputted + 1}/{file_count}")
            image = Image.fromarray((entry * 255).astype(np.uint8))
            image.save(f"output/{outputted:05d}.png")
            outputted += 1
            if outputted == file_count:
                return


def main():
    args = parse_args()

    configure_tensorflow()

    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.patience)

    if args.out:
        output(args.file_count, args.batch_size)


if __name__ == "__main__":
    main()
