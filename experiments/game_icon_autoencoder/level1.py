import os
import sys
import numpy as np

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
    BaseModel,
    add_pool_convolution_2d,
    add_fully_connected,
    add_upsampling_convolution_2d,
    add_convolution_2d,
)

from PIL import Image


basepath = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.abspath(os.path.join(basepath, "../data/game_icons"))
sys.path.append(datapath)
from dataset import get_dataset


dataset = get_dataset()


INPUT_SHAPE = (128, 128, 3)
# VALIDATION_DATA_NAME = "data-level1-validation.npy"
# TRAINING_DATA_NAME = "data-level1-training.npy"
# FILES_DATA_NAME = "data-files.npy"


# def get_files_data():
#     return np.load(file=get_input_path(FILES_DATA_NAME))


def get_training_data():
    return dataset.training_data


def get_validation_data():
    return dataset.validation_data


class Level1Model(BaseModel):

    def build(self):

        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))

        encoder = Sequential(name="encoder")
        add_pool_convolution_2d(encoder, 32)  # TODO: Make prelu
        add_pool_convolution_2d(encoder, 64)
        add_pool_convolution_2d(encoder, 128)
        add_pool_convolution_2d(encoder, 256)
        encoder.add(Flatten())
        add_fully_connected(encoder, 128)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        add_fully_connected(decoder, 8 * 8 * 256)
        decoder.add(Reshape((8, 8, 256)))
        add_upsampling_convolution_2d(decoder, 256)
        add_upsampling_convolution_2d(decoder, 128)
        add_upsampling_convolution_2d(decoder, 64)
        add_upsampling_convolution_2d(decoder, 32)
        add_convolution_2d(decoder, 3)
        model.add(decoder)

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
        decoder.add(InputLayer(input_shape=(128,), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)


# def plot():
#     import matplotlib.pyplot as plt
#     files = get_files_data()
#     model = Level1Model()
#     plt.subplot(2, 2, 1)
#     plt.plot(files[0][200])

#     plt.subplot(2, 2, 2)
#     plt.plot(model.predict_output(files[0][200].reshape(1, INPUT_COUNT, 1)).flatten())

#     plt.subplot(2, 2, 3)
#     plt.plot(files[0][210])

#     plt.subplot(2, 2, 4)
#     plt.plot(model.predict_output(files[0][210].reshape(1, INPUT_COUNT, 1)).flatten())

#     plt.show()


def train(batch_size, epochs, verbose, data_percentage, patience):
    model = Level1Model()

    print(f"Loading training data")
    training_data = get_training_data()
    training_data = training_data[:int(len(training_data) * data_percentage)]
    validation_data = get_validation_data()
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


def output(file_count):
    model = Level1Model()
    # inp = get_validation_data()[:file_count]
    import random
    # inp = np.array([[random.random() for _ in range(128)] for _ in range(file_count)])

    inputs = get_validation_data()
    encoded = model.encode(inputs)
    merged = []

    for i in range(file_count):
        a = random.choice(encoded)
        b = random.choice(encoded)
        # vals = []
        # for i in range(128):
        #     vals.append(enc[i] if random.random() > 0.5 else pair[i])
        # merged.append(vals)
        merged.append(a + b / 2)
    merged = np.array(merged)

    output = model.decode(merged)
    for index, out in enumerate(output):
        out = (out * 255).astype(np.uint8)
        image = Image.fromarray(out)
        image.save(f"output-{index}.png")


def main():
    configure_tensorflow()
    args = parse_args()

    # if args.format_data:
    #     format_data(args.file_count)
    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    # if args.plot:
    #     plot()
    if args.out:
        output(args.file_count)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nExiting early")
