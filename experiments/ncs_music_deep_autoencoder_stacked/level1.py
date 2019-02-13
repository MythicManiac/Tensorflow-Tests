import os
import sys

import numpy as np

from keras.layers import (
    InputLayer,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Activation,
)
from keras.models import Sequential

from common import (
    configure_tensorflow,
    parse_args,
    get_input_path,
    get_output_path,
    BaseModel,
    write_wav,
)


INPUT_COUNT = 16384
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

        def add_convolution(sequential, filters):
            sequential.add(Conv1D(filters, 3, padding="same", use_bias=False))
            sequential.add(BatchNormalization())
            sequential.add(Activation("relu"))

        # encoder = Sequential(name="encoder")
        add_convolution(model, 16)
        model.add(MaxPool1D(2))
        add_convolution(model, 32)
        model.add(MaxPool1D(2))
        add_convolution(model, 32)
        model.add(MaxPool1D(2))
        add_convolution(model, 64)
        model.add(MaxPool1D(2))
        add_convolution(model, 128)
        model.add(MaxPool1D(2, name="latent"))
        # model.add(encoder)

        # decoder = Sequential(name="decoder")
        add_convolution(model, 128)
        model.add(UpSampling1D(2))
        add_convolution(model, 64)
        model.add(UpSampling1D(2))
        add_convolution(model, 32)
        model.add(UpSampling1D(2))
        add_convolution(model, 32)
        model.add(UpSampling1D(2))
        add_convolution(model, 16)
        model.add(UpSampling1D(2))
        # add_convolution(model, 1)
        model.add(Conv1D(1, 3, name="out", padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(INPUT_COUNT, 1), name="in"))
        encoder.add(self.model.get_layer("conv1d_1"))
        encoder.add(self.model.get_layer("batch_normalization_1"))
        encoder.add(self.model.get_layer("activation_1"))
        encoder.add(self.model.get_layer("max_pooling1d_1"))
        encoder.add(self.model.get_layer("conv1d_2"))
        encoder.add(self.model.get_layer("batch_normalization_2"))
        encoder.add(self.model.get_layer("activation_2"))
        encoder.add(self.model.get_layer("max_pooling1d_2"))
        encoder.add(self.model.get_layer("conv1d_3"))
        encoder.add(self.model.get_layer("batch_normalization_3"))
        encoder.add(self.model.get_layer("activation_3"))
        encoder.add(self.model.get_layer("max_pooling1d_3"))
        encoder.add(self.model.get_layer("conv1d_4"))
        encoder.add(self.model.get_layer("batch_normalization_4"))
        encoder.add(self.model.get_layer("activation_4"))
        encoder.add(self.model.get_layer("max_pooling1d_4"))
        encoder.add(self.model.get_layer("conv1d_5"))
        encoder.add(self.model.get_layer("batch_normalization_5"))
        encoder.add(self.model.get_layer("activation_5"))
        encoder.add(self.model.get_layer("latent"))
        encoder.summary()
        self.encoder = encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(512, 128), name="in"))
        decoder.add(self.model.get_layer("conv1d_6"))
        decoder.add(self.model.get_layer("batch_normalization_6"))
        decoder.add(self.model.get_layer("activation_6"))
        decoder.add(self.model.get_layer("up_sampling1d_1"))
        decoder.add(self.model.get_layer("conv1d_7"))
        decoder.add(self.model.get_layer("batch_normalization_7"))
        decoder.add(self.model.get_layer("activation_7"))
        decoder.add(self.model.get_layer("up_sampling1d_2"))
        decoder.add(self.model.get_layer("conv1d_8"))
        decoder.add(self.model.get_layer("batch_normalization_8"))
        decoder.add(self.model.get_layer("activation_8"))
        decoder.add(self.model.get_layer("up_sampling1d_3"))
        decoder.add(self.model.get_layer("conv1d_9"))
        decoder.add(self.model.get_layer("batch_normalization_9"))
        decoder.add(self.model.get_layer("activation_9"))
        decoder.add(self.model.get_layer("up_sampling1d_4"))
        decoder.add(self.model.get_layer("conv1d_10"))
        decoder.add(self.model.get_layer("batch_normalization_10"))
        decoder.add(self.model.get_layer("activation_10"))
        decoder.add(self.model.get_layer("up_sampling1d_5"))
        decoder.add(self.model.get_layer("out"))
        decoder.add(self.model.get_layer("batch_normalization_11"))
        decoder.add(self.model.get_layer("activation_11"))
        decoder.summary()
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
    print(f"Beginning training with {batch_size} batch size across {epochs} epochs")
    print(f"Early stop patience: {patience}")
    training_data, validation_data = get_training_data()
    training_cutoff = int(len(training_data) * data_percentage)
    validation_cutoff = int(len(validation_data) * data_percentage)
    model = Level1Model()
    model.train(
        in_x=training_data[:training_cutoff],
        in_y=training_data[:training_cutoff],
        val_x=validation_data[:validation_cutoff],
        val_y=validation_data[:validation_cutoff],
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


def format_data():
    basepath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.abspath(os.path.join(basepath, "../data/ncs_music"))
    sys.path.append(datapath)
    from highfreq_dataset import get_dataset
    data = get_dataset(
        block_interval=int(INPUT_COUNT / 2),
        block_size=INPUT_COUNT,
        file_count=107,
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
        format_data()
    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.plot:
        plot()
    if args.out:
        output(args.file_count)


if __name__ == "__main__":
    main()
