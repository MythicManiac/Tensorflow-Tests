import numpy as np
import tensorflow as tf

from data.iris_flower.dataset import get_dataset


class FeedForwardNeuralNetwork(object):

    def __init__(self):
        self.features = 4
        self.labels = 3
        self.hidden = 10
        self.learning_rate = 0.005
        self.build()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        tf.summary.FileWriter("logs/", self.session.graph)

    def build(self):
        # None in the shape indicates that axis can be any size
        # in this case it means we can provide input in any batch size
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.features), name="inputs")
        self.output = tf.placeholder(tf.float32, shape=(None, self.labels), name="labels")

        weights_1 = tf.Variable(tf.truncated_normal([self.features, self.hidden]), name="w1")
        weights_2 = tf.Variable(tf.truncated_normal([self.hidden, self.labels]), name="w2")
        # Truncated normal is like random normal, but cuts off if the value
        # is too big/small

        bias_1 = tf.Variable(tf.zeros([self.hidden]), name="b1")
        bias_2 = tf.Variable(tf.zeros([self.labels]), name="b2")

        logits_1 = tf.matmul(self.inputs, weights_1) + bias_1
        relu_1 = tf.nn.relu(logits_1)
        # tf.nn.relu (REctified Linear Unit) is an activation function that returns
        # a value of 0 or above. It can be defined as f(x) = max(0, x)

        logits_2 = tf.matmul(relu_1, weights_2) + bias_2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_2, labels=self.output))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.prediction = tf.nn.softmax(logits_2)

    def get_acccuracy(self, predictions, labels):
        return (
            100.0 *
            np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            len(predictions)
        )

    def train(self, step_count, inputs, outputs):
        for step in range(step_count):
            _, loss, predictions = self.session.run(
                [self.optimizer, self.loss, self.prediction],
                feed_dict={
                    self.inputs: inputs,
                    self.output: outputs,
                }
            )

            if step % 100 == 0:
                print("-" * 30)
                print(f"Step {step}")
                print("-" * 30)
                print(f"Loss: {loss}")
                print(f"Accuracy: {self.get_acccuracy(predictions, outputs)}")

    def test(self, inputs, outputs):
        predictions = self.session.run(
            self.prediction,
            feed_dict={
                self.inputs: inputs,
                self.output: outputs,
            }
        )
        print("-" * 30)
        print(f"Test results")
        print("-" * 30)
        print(f"Accuracy: {self.get_acccuracy(predictions, outputs)}")


def main():
    dataset = get_dataset()
    print(dataset.train_data_in.shape)
    print(dataset.train_data_labels.shape)

    model = FeedForwardNeuralNetwork()
    model.train(2000, dataset.train_data_in, dataset.train_data_labels)
    model.test(dataset.test_data_in, dataset.test_data_labels)


if __name__ == "__main__":
    main()
