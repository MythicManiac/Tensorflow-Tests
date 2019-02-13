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
from keras import backend

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


INPUT_SHAPE = (1024, 16)
VALIDATION_DATA_NAME = "data-level2-validation.npy"
TRAINING_DATA_NAME = "data-level2-training.npy"


def reshape_data(data):
    data = data[:int(len(data) / INPUT_SHAPE[0]) * INPUT_SHAPE[0]]
    assert len(data) % INPUT_SHAPE[0] == 0
    data = np.array(np.split(data, int(len(data) / INPUT_SHAPE[0])))
    data = data.reshape(len(data), *INPUT_SHAPE)
    return data


def get_training_data():
    data = np.load(file=get_input_path(TRAINING_DATA_NAME), allow_pickle=False)
    return data


def get_validation_data():
    data = np.load(file=get_input_path(VALIDATION_DATA_NAME), allow_pickle=False)
    return data


class Level2Model(BaseModel):

    def build(self):

        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name="in"))

        encoder = Sequential(name="encoder")
        add_pool_convolution(encoder, 16)
        add_pool_convolution(encoder, 32)
        add_pool_convolution(encoder, 64)
        add_pool_convolution(encoder, 128)
        add_pool_convolution(encoder, 128)
        model.add(encoder)

        decoder = Sequential(name="decoder")
        add_upsampling_convolution(decoder, 128)
        add_upsampling_convolution(decoder, 128)
        add_upsampling_convolution(decoder, 64)
        add_upsampling_convolution(decoder, 32)
        add_upsampling_convolution(decoder, 16)
        add_convolution(decoder, INPUT_SHAPE[1])
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
        decoder.add(InputLayer(input_shape=(1, 128), name="in"))
        decoder.add(self.model.get_layer("decoder"))
        self.decoder = decoder


def plot():
    import matplotlib.pyplot as plt
    model = Level2Model()
    files = get_validation_data()
    print(files.shape)

    def plot_single(rows, columns, index, entry):
        plt.subplot(rows, columns, index)
        plt.plot(entry)

        entry = entry.reshape(1, *entry.shape)
        result = model.predict_output(entry)

        plt.subplot(rows, columns, index + 1)
        plt.plot(result[0])

    plot_single(2, 2, 1, files[0])
    plot_single(2, 2, 3, files[1])

    plt.show()


def train(batch_size, epochs, verbose, data_percentage, patience):
    print(f"Beginning training with {batch_size} batch size across {epochs} epochs")
    print(f"Early stop patience: {patience}")
    training_data = get_training_data()
    validation_data = get_validation_data()

    training_data = training_data[:int(len(training_data) * data_percentage)]
    validation_data = validation_data[:int(len(validation_data) * data_percentage)]

    model = Level2Model()
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
    import level1

    files = level1.get_files_data()
    file = files[0]
    file = file.reshape(len(file), level1.INPUT_COUNT, 1)
    del files

    model = level1.Level1Model()
    file = model.encode(file)
    del model
    backend.clear_session()

    model = Level2Model()
    file = model.predict_output(file)
    del model
    backend.clear_session()

    model = level1.Level1Model()
    file = file.reshape(file.shape[0] * file.shape[1], 1, 16)
    file = model.decode(file)
    del model
    backend.clear_session()

    write_wav(get_output_path(f"output-level2-0.wav"), file.flatten())


def format_data(data_percentage):
    import level1
    model = level1.Level1Model()

    training = level1.get_training_data()
    training_cutoff = int(len(training) * data_percentage)
    training = training[:training_cutoff]

    print(f"Training data: {training.shape}")
    out = model.encode(training)
    print(f"Training data level 2: {out.shape}")
    np.save(file=get_output_path(TRAINING_DATA_NAME), arr=out, allow_pickle=False)
    del out
    del training

    validation = level1.get_validation_data()
    validation_cutoff = int(len(validation) * data_percentage)
    validation = validation[:validation_cutoff]

    print(f"Validation data: {validation.shape}")
    out = model.encode(validation)
    print(f"Validation data level 2: {out.shape}")
    np.save(file=get_output_path(VALIDATION_DATA_NAME), arr=out, allow_pickle=False)
    del out
    del validation


def main():
    configure_tensorflow()
    args = parse_args()

    if args.format_data:
        format_data(args.data_percentage)
    if args.train:
        train(args.batch_size, args.epochs, args.verbose, args.data_percentage, args.patience)
    if args.plot:
        plot()
    if args.out:
        output(args.file_count)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nExiting early")
