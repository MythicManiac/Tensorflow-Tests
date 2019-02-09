import os
import wave

import numpy as np


class Dataset(object):

    def get_path(self, path):
        basepath = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(basepath, path)
        return csv_path

    def get_all_files(self, path):
        path = self.get_path(path)
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        return files

    def read_wav(self, path, block_size):
        with wave.open(path, "rb") as f:
            nchannels, sampwidth, framerate, nframes, comptype, compname = f.getparams()
            frames = f.readframes(nframes)

        # Make sure we have an uncompressed, mono WAV, with 1 byte samples and 4000 frames per second
        # This can be upscaled later on, but for now it reduces required parameter count
        assert comptype == "NONE"
        assert compname == "not compressed"
        assert sampwidth == 1
        assert nchannels == 1
        assert framerate == 44100
        assert nframes == len(frames)

        frames = frames[:int(len(frames) / block_size) * block_size]
        assert len(frames) % block_size == 0

        # print(f"Read wav file at {path}")
        # print("-" * 20)
        # print(f"Channel count: {nchannels}")
        # print(f"Sample width: {sampwidth}")
        # print(f"Frame rate: {framerate}")
        # print(f"Frame count: {nframes}")
        # print(f"Comp type: {comptype}")
        # print(f"Comp name: {compname}")
        # print(f"Data count: {len(frames)}")

        # Normalize
        frames = np.array(np.split(np.array(list(frames)), len(frames) / block_size)) / 255.0

        return frames

    def write_wav(self, path, data):
        params = (
            1,                 # Channel count
            1,                 # Sample width
            44100,             # Frame rate
            len(data),         # Frame count
            "NONE",            # Compression type
            "not compressed",  # Compression name
        )

        # Convert back to bytes
        data = bytes((data * 255).astype(np.byte))

        with wave.open(path, "wb") as f:
            f.setparams(params)
            f.writeframes(data)

    def __init__(self):
        pass

    def load(self, block_size=800, block_interval=400, file_count=None,
             shuffle=True, output_size=1, just_files=False):
        files = self.get_all_files("highfreq")
        if file_count is None:
            file_count = len(files)
        file_count = min(len(files), file_count)
        data = []
        for i in range(file_count):
            print(f"Processing file {i + 1} / {file_count}")
            wav_data = self.read_wav(
                files[i],
                block_size=block_size,
            )
            data.append(wav_data)
        self.files = np.array(data)

        if just_files:
            print(f"Loaded {len(self.files)} files")
            return

        data = []
        outputs = []
        for file in self.files:
            full_data = file.flatten()
            pos = 0
            while pos + block_size < len(full_data) and pos + block_size + output_size < len(full_data):
                data.append(full_data[pos: pos + block_size])
                if output_size:
                    outputs.append(full_data[pos + block_size:pos + block_size + output_size])
                pos += block_interval

        # def unison_shuffled_copies(a, b):
        #     assert len(a) == len(b)
        #     p = np.random.permutation(len(a))
        #     return a[p], b[p]

        # if shuffle:
        #     data, outputs = unison_shuffled_copies(data, outputs)

        if shuffle:
            outputs = []
            np.random.shuffle(data)

        cutoff = int(len(data) * 0.8)
        self.train_data = np.array(data[:cutoff])
        self.train_out = np.array(outputs[:cutoff])
        self.test_data = np.array(data[cutoff:])
        self.test_out = np.array(outputs[cutoff:])
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")


def get_dataset(file_count=None, block_size=800, block_interval=400, shuffle=True, output_size=1, just_files=False):
    dataset = Dataset()
    dataset.load(
        file_count=file_count,
        block_size=block_size,
        block_interval=block_interval,
        shuffle=shuffle,
        output_size=output_size,
        just_files=just_files,
    )
    return dataset
