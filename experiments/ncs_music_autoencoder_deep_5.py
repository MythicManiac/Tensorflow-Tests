import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Input,
    InputLayer,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Activation,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Model, Sequential, load_model
from keras import backend

from data.ncs_music.highfreq_dataset import get_dataset


MODEL_ID = 5
NAME = "autoencoder_deep"
SAVE_LOCATION = f"models/ncs_music_autoencoder_deep_{MODEL_ID}.h5"
INPUT_COUNT = 16384
OUTPUT_COUNT = INPUT_COUNT


class ExperimentalModel(object):

    def __init__(self):
        self.n_inputs = INPUT_COUNT
        self.n_outputs = OUTPUT_COUNT
        self._decoder = None
        self.build()

    def build(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))

        # encoder = Sequential(name="encoder")
        model.add(Conv1D(16, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(2))
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(2))
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(2))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool1D(2))
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPool1D(2, name="latent"))
        # model.add(encoder)

        # decoder = Sequential(name="decoder")
        model.add(Conv1D(128, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(2))
        model.add(Conv1D(64, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(2))
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(2))
        model.add(Conv1D(32, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(2))
        model.add(Conv1D(16, 3, padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling1D(2))
        model.add(Conv1D(1, 3, name="out", padding="same", use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # model.add(decoder)

        model.summary()
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        self.model = model

        self.build_encoder()

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(self.n_inputs, 1), name="in"))
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

    def load(self):
        if os.path.exists(SAVE_LOCATION):
            self.model.load_weights(SAVE_LOCATION)
            print("Loaded a model")

    def train(self, in_x, in_y, val_x, val_y):
        model_checkpoint = ModelCheckpoint(
            SAVE_LOCATION,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0.0001
        )
        # early_stoppping = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0,
        #     patience=5,
        #     verbose=1,
        #     mode="auto"
        # )
        self.model.fit(
            in_x,
            in_y,
            epochs=3000,
            batch_size=128,
            validation_data=(val_x, val_y),
            callbacks=[model_checkpoint, reduce_lr],
        )
        # self.model.save(SAVE_LOCATION)

    def decode(self, inputs):
        return self.decoder.predict(inputs)

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def predict_output(self, inputs):
        return self.model.predict(inputs)


def main():
    # backend.set_floatx("float16")
    # backend.set_epsilon(1e-4)

    data = get_dataset(
        block_interval=10000,
        block_size=INPUT_COUNT,
        file_count=1,
        output_size=0,
        shuffle=True,
    )
    train_data = data.train_data.reshape(len(data.train_data), INPUT_COUNT, 1)
    test_data = data.test_data.reshape(len(data.test_data), INPUT_COUNT, 1)

    model = ExperimentalModel()
    model.load()

    if "--train" in sys.argv:
        model.train(train_data, train_data, test_data, test_data)

    if "--plot" in sys.argv:
        plt.subplot(2, 2, 1)
        plt.plot(data.files[0][200])

        plt.subplot(2, 2, 2)
        plt.plot(model.predict_output(data.files[0][200].reshape(1, INPUT_COUNT, 1)).flatten())

        plt.subplot(2, 2, 3)
        plt.plot(data.files[0][210])

        plt.subplot(2, 2, 4)
        plt.plot(model.predict_output(data.files[0][210].reshape(1, INPUT_COUNT, 1)).flatten())

        plt.show()

    if "--out" in sys.argv:
        for i in range(min(len(data.files), 10)):
            inp = data.files[i].reshape(len(data.files[i]), INPUT_COUNT, 1)
            output = model.predict_output(inp).flatten()
            data.write_wav(f"output-{NAME}-{MODEL_ID}-{i}.wav", output)
            print(f"output-{NAME}-{MODEL_ID}-{i}.wav created")

    if "--vis" in sys.argv:
        os.environ["FFMPEG_BINARY"] = "ffmpeg"
        from moviepy.editor import (
            CompositeVideoClip, AudioFileClip, VideoClip
        )

        file = data.files[0]
        inp = file.reshape(len(file), INPUT_COUNT, 1)
        output = model.encode(inp)
        output = output.reshape(output.shape[0] * 512, 128)
        min_val = np.amin(output)
        max_val_normalized = np.amax(output) - min_val

        class Figure(object):

            def __init__(self, width, height, row, column, frame_duration):
                self.width = width
                self.height = height
                self.row = row
                self.column = column
                self.frame_duration = frame_duration
                self.current_highest = 0
                self.buffer = [0 for i in range(self.width)]
                self.figures = []

            def push(self, val):
                if val > self.buffer[-1]:
                    self.buffer[-1] = val

            def render(self, peaks):
                figure = np.zeros((self.width, self.height), int)
                for column, peak in enumerate(peaks):
                    for fill in range(int(round(peak * (self.height - 1)))):
                        figure[
                            self.height - 1 - fill,
                            column
                        ] = 255
                return np.stack((figure,) * 3, axis=-1)

            def snapshot(self):
                self.figures.append(self.buffer)
                self.buffer = self.buffer[1:self.width] + [0]

        class FigureClip(VideoClip):

            def __init__(self, figure):
                super().__init__()
                self.figure = figure
                self.make_frame = lambda time: self.make_into_frame(time)
                self.start = 0
                self.end = figure.frame_duration * len(figure.figures)
                self.size = (figure.width, figure.height)

                # 16 columns
                # 8 rows
                # padding of 6px
                self.pos = lambda _: (
                    66 * figure.column + 6,
                    66 * figure.row + 6
                )

            def make_into_frame(self, time):
                index = int(time / self.figure.frame_duration)
                if index > len(self.figure.figures):
                    return np.zeros(self.figure.width, self.figure.height)
                return self.figure.render(self.figure.figures[index])

        last_percentage = -1
        figures = []

        # (graph total duration / graph datapoint count) * (graph datapoint count / graph width)
        figure_snapshot_rate = 40
        tick_to_sample_ratio = 32.87890625
        frame_duration = (figure_snapshot_rate * tick_to_sample_ratio) / 44100
        for i in range(128):
            column = i % 16
            row = int(i / 16)
            figures.append(Figure(60, 60, row, column, frame_duration))

        print(f"Rendering output: {output.shape}")
        for index, entry in enumerate(output):
            should_snapshot = index % figure_snapshot_rate == 0

            for plot_index, plot in enumerate(figures):
                plot.push((entry[plot_index] - min_val) / max_val_normalized)

                if should_snapshot:
                    plot.snapshot()

            percentage = int(index / len(output) * 100)
            if percentage % 1 == 0 and last_percentage != percentage:
                last_percentage = percentage
                print(f"Capturing figures: {percentage}%...")

        print(f"{len(figures[0].figures)} figure frames rendered")
        clips = [FigureClip(figure) for figure in figures]

        audio_filename = f"vis/output.wav"
        output = model.predict_output(inp).flatten()
        data.write_wav(audio_filename, output)

        del model
        backend.clear_session()

        audio = AudioFileClip(audio_filename)
        audio = audio.set_start(0)
        audio = audio.set_duration(min(audio.duration, frame_duration * len(figures[0].figures)))

        result = CompositeVideoClip(clips, size=(16 * 66 + 12, 8 * 66 + 12))
        result = result.set_audio(audio)
        result.write_videofile("vis/output.mp4", fps=1 / frame_duration)


if __name__ == "__main__":
    main()
