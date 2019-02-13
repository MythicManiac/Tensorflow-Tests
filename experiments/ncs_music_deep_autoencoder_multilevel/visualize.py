import os

import numpy as np

from keras import backend

from moviepy.editor import (
    CompositeVideoClip, AudioFileClip, VideoClip
)

from common import write_wav


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


def visualize(model_cls, input_data):
    os.environ["FFMPEG_BINARY"] = "ffmpeg"

    model = model_cls()
    output = model.encode(input_data)
    output = output.reshape(output.shape[0] * 512, 128)
    min_val = np.amin(output)
    max_val_normalized = np.amax(output) - min_val

    last_percentage = -1
    figures = []

    # (graph total duration / graph datapoint count) * (graph datapoint count / graph width)
    figure_snapshot_rate = 40
    tick_to_sample_ratio = 32.87890625  # This is still off sync with the audio, 2:53 becomes 2:58 for some reason
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
    output = model.predict_output(input_data).flatten()
    write_wav(audio_filename, output)

    del model
    backend.clear_session()

    audio = AudioFileClip(audio_filename)
    audio = audio.set_start(0)
    audio = audio.set_duration(min(audio.duration, frame_duration * len(figures[0].figures)))

    result = CompositeVideoClip(clips, size=(16 * 66 + 12, 8 * 66 + 12))
    result = result.set_audio(audio)
    result.write_videofile("vis/output.mp4", fps=1 / frame_duration)
