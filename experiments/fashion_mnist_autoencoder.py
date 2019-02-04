import os

import tensorflow as tf

from data.fashion_mnist.dataset import get_dataset


CHECKPOINT_LOCATION = "models/fashion_mnist_autoencoder.ckpt"


class AutoEncoder(object):

    def __init__(self, learning_rate=0.05):
        self.features = 784
        self.hidden = 32
        self.learning_rate = learning_rate
        self.build()
        self.config = tf.ConfigProto(
            # device_count={
            #     "GPU": 0
            # }
        )
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.session = tf.Session(config=self.config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if os.path.isfile(CHECKPOINT_LOCATION):
            self.saver.restore(self.session, CHECKPOINT_LOCATION)
        tf.summary.FileWriter("logs/", self.session.graph)

    def build(self):
        # Our network has 3 connection layers:
        #   input nodes x 784                 |
        #     784 x 32 connections to         |-- Encoder
        #   hidden nodes x 32                 |
        #     32 x 32 connections to            -- Bridge
        #   hidden nodes x 32                 |
        #     32 x 784 connections to         |-- Decoder
        #   output nodes x 784                |

        # Define the variables and biases of the network layers
        encoder_hidden_values = {
            "weights": tf.Variable(tf.random_normal([self.features, self.hidden]), name="encoder_w"),
            "biases": tf.Variable(tf.random_normal([self.hidden]), name="encoder_b")
        }

        bridge_hidden_values = {
            "weights": tf.Variable(tf.random_normal([self.hidden, self.hidden]), name="bridge_w"),
            "biases": tf.Variable(tf.random_normal([self.hidden]), name="bridge_b")
        }

        decoder_hidden_values = {
            "weights": tf.Variable(tf.random_normal([self.hidden, self.features]), name="decoder_w"),
            "biases": tf.Variable(tf.random_normal([self.features]), name="decoder_b")
        }

        # None = any size
        self.inputs = tf.placeholder(tf.float32, shape=(None, 784))

        encoder_layer = tf.nn.sigmoid(
            tf.add(
                tf.matmul(self.inputs, encoder_hidden_values["weights"]),
                encoder_hidden_values["biases"]))

        bridge_layer = tf.nn.sigmoid(
            tf.add(
                tf.matmul(encoder_layer, bridge_hidden_values["weights"]),
                bridge_hidden_values["biases"]))

        decoder_layer = tf.nn.sigmoid(
            tf.add(
                tf.matmul(bridge_layer, decoder_hidden_values["weights"]),
                decoder_hidden_values["biases"]))

        self.output = decoder_layer
        self.reference_output = tf.placeholder(tf.float32, shape=(None, 784))

        self.loss = tf.reduce_mean(tf.square(self.output - self.reference_output))
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_data, batch_size, epochs):
        batches = int(len(input_data) / batch_size)
        loss = 0

        for epoch in range(epochs):
            for batch in range(batches):
                # print(f"Running batch {batch + 1} / {batches}")
                batch_index = batch * batch_size
                batch_input = input_data[batch_index: batch_index + batch_size]
                _, loss = self.session.run(
                    [self.optimizer, self.loss],
                    feed_dict={
                        self.inputs: batch_input,
                        self.reference_output: batch_input,
                    }
                )

            print("-" * 30)
            print(f"Epoch {epoch + 1} / {epochs} | Loss: {loss}")
            print("-" * 30)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Saved model to {CHECKPOINT_LOCATION}")
                self.saver.save(self.session, CHECKPOINT_LOCATION)


def main():
    data = get_dataset()
    network = AutoEncoder(learning_rate=0.1)
    network.train(input_data=data.train_data, batch_size=100, epochs=1000)


if __name__ == "__main__":
    main()
