import os
import sys

import numpy as np

from keras.layers import (
    InputLayer,
)
from keras.models import Sequential

from common import (
    configure_tensorflow,
    parse_args,
    get_input_path,
    get_output_path,
    BaseModel,
    write_wav,
    add_pool_convolution,
    add_upsampling_convolution,
    add_convolution,
)


INPUT_COUNT = 1024
VALIDATION_DATA_NAME = "data-level1-validation.npy"
TRAINING_DATA_NAME = "data-level1-training.npy"
FILES_DATA_NAME = "data-files.npy"


def get_files_data():
    return np.load(file=get_input_path(FILES_DATA_NAME))


def get_training_data():
    return (
        np.load(file=get_input_path(TRAINING_DATA_NAME), allow_pickle=False),
        np.load(file=get_input_path(VALIDATION_DATA_NAME), allow_pickle=False),
    )


class Level1Model(BaseModel):

    def build(self):

        model = Sequential()
        model.add(InputLayer(input_shape=(INPUT_COUNT, 1), name="in"))

        encoder = Sequential(name="encoder")
        add_pool_convolution(encoder, 4)
        add_pool_convolution(encoder, 4)
        add_pool_convolution(encoder, 8)
        add_pool_convolution(encoder, 8)
        add_pool_convolution(encoder, 32)
        add_pool_convolution(encoder, 4)
        add_pool_convolution(encoder, 4)
        add_pool_convolution(encoder, 8)
        add_pool_convolution(encoder, 8)
        add_pool_convolution(encoder, 128)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        add_upsampling_convolution(decoder, 128)
        add_upsampling_convolution(decoder, 8)
        add_upsampling_convolution(decoder, 8)
        add_upsampling_convolution(decoder, 4)
        add_upsampling_convolution(decoder, 4)
        add_upsampling_convolution(decoder, 32)
        add_upsampling_convolution(decoder, 8)
        add_upsampling_convolution(decoder, 8)
        add_upsampling_convolution(decoder, 4)
        add_upsampling_convolution(decoder, 4)
        add_convolution(decoder, 1)
        model.add(decoder)

        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(INPUT_COUNT, 1), name="in"))
        encoder.add(self.model.get_layer("encoder"))
        self.encoder = encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(1, 128), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, inputs):
        return self.decoder.predict(inputs)


def plot():
    import matplotlib.pyplot as plt
    files = get_files_data()
    model = Level1Model()
    plt.subplot(2, 2, 1)
    plt.plot(files[0][200])

    plt.subplot(2, 2, 2)
    plt.plot(model.predict_output(files[0][200].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.subplot(2, 2, 3)
    plt.plot(files[0][210])

    plt.subplot(2, 2, 4)
    plt.plot(model.predict_output(files[0][210].reshape(1, INPUT_COUNT, 1)).flatten())

    plt.show()


def train(batch_size, epochs, verbose, data_percentage, patience):
    model = Level1Model()

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


def output(file_count):
    files = get_files_data()
    model = Level1Model()
    for i in range(min(len(files), file_count)):
        inp = files[i].reshape(len(files[i]), INPUT_COUNT, 1)
        output = model.predict_output(inp).flatten()
        write_wav(get_output_path(f"output-level1-{i}.wav"), output)


def format_data(file_count):
    basepath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.abspath(os.path.join(basepath, "../data/ncs_music"))
    sys.path.append(datapath)
    from highfreq_dataset import get_dataset
    data = get_dataset(
        block_interval=int(INPUT_COUNT / 2),
        block_size=INPUT_COUNT,
        file_count=file_count,
        output_size=0,
        shuffle=True,
        just_files=False,
    )
    data.train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1).astype(np.float16)
    data.test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1).astype(np.float16)
    np.save(file=get_output_path(TRAINING_DATA_NAME), arr=data.train_data, allow_pickle=False)
    np.save(file=get_output_path(VALIDATION_DATA_NAME), arr=data.test_data, allow_pickle=False)
    np.save(file=get_output_path(FILES_DATA_NAME), arr=data.files)


def main():
    configure_tensorflow()
    args = parse_args()

    if args.format_data:
        format_data(args.file_count)
    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.plot:
        plot()
    if args.out:
        output(args.file_count)


if __name__ == "__main__":
    main()
