import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from encoding import BinarySNN
from lif_neuron import LIFNeuron


class MnistSNN(BinarySNN):

    def __init__(self, train_features, train_labels, validation_features, validation_labels, test_features,
                 test_labels):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.validation_features = validation_features
        self.validation_labels = validation_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.n_labels = len(self.train_labels[0])
        self.input_length = len(self.train_features[0])
        self.weights = np.zeros((self.n_labels, self.input_length))
        self.time = np.arange(0, self.total_time, 1)

        self.sliding_window = np.arange(-5, 5, 1)
        self.current_constant = 20
        self.network = []
        self.weight_init()
        self.op_neurons = [LIFNeuron() for _ in range(self.n_labels)]

    def weight_init(self):
        for i in range(self.n_labels):
            for j in range(self.input_length):
                self.weights[i][j] = 0  # np.random.uniform(0, 1)

    def spikes_to_current(self, spike_train, weights):
        """
        Converts spike train to current using weights
        :param spike_train: Spike train for which current needs to be generated
        :param weights: The weights associated with that synapse
        :return: The current associated with the spike train and the weights
        """
        return (sum(spike_train) / len(spike_train)) * weights * self.current_constant

    def test_step_mnist(self, input_trains):
        """
        For a set of input spike trains representing an image, this function predicts the digit
        :param input_trains: The input spike trains each representing one image
        :return: A list of digits predicted from the input spike trains
        """
        outputs = []
        for inp in input_trains:
            neuron_spikes = []
            for j in range(self.n_labels):
                self.op_neurons[j].reset()
                current = 0
                for i in range(len(inp)):
                    current += self.spikes_to_current(inp[i], self.weights[j][i])
                time_values = list(np.linspace(1, self.total_time, self.total_time))
                potentials_list, spike_count = self.op_neurons[j].simulate_neuron(current, time_values, 1)
                neuron_spikes.append(spike_count)
            outputs.append(neuron_spikes.index(max(neuron_spikes)))
        return outputs

    def train_mnist(self, input_trains, output_trains):
        """
        Trains the weights of the network using the input spike trains and the corresponding expected output spike
        trains. The weights are stored as class variables so this function can be called independently for each sample.
        :param input_trains: The input train representing an image
        :param output_trains: The expected output trains corresponding to the input image.
        :return:
        """
        for t in self.time:
            for j in range(self.n_labels):
                if output_trains[j][t] == 1:
                    for i in range(self.input_length):
                        for t1 in self.sliding_window:
                            if 0 <= t + t1 < self.total_time and t1 != 0 and input_trains[i][t + t1] == 1:
                                self.weights[j][i] = self.update_weights(self.weights[j][i], t1)

    @staticmethod
    def one_hot_encode_mnist(labels):
        """
        The labels are one hot encoded.
        :param labels: The input labels that need to be encoded.
        :return: The one hot encoded labels. Eg: 0 -> [1,0,0,0,0,0,0,0,0,0]
        """
        encoded_op = [[0 for _ in range(10)] for _ in labels]
        for i, l in enumerate(labels):
            encoded_op[i][l] = 1
        return encoded_op

    @staticmethod
    def normalize_and_flatten_images(images):
        """
        Images are normalized and flattened. Normalization is done as follows: If the grayscale
        value of a pixel is greater than 8, then the pixel is given a value of 1 else it is given 0.
        :param images: Raw images with 0-15 grayscale values
        :return: Normalized and flattened images
        """
        normalized_images = []
        for im in images:
            normalized_image = np.zeros((len(images[0]), len(images[0][0])))
            for j in range(len(im)):
                for k in range(len(im[j])):
                    if im[j][k] > 8:
                        normalized_image[j][k] = 1
            normalized_images.append(normalized_image.flatten())
        return normalized_images

    def normalize_weights(self):
        """
        Normalizes weights according to (w - min_w)/(max_w - min_w)
        """
        flattened = self.weights.flatten()
        max_w = max(flattened)
        min_w = min(flattened)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] = (self.weights[i][j] - min_w) / (max_w - min_w)

    @staticmethod
    def data_extraction(sub_data=None):
        """
        Loads the digits data into appropriate data structures and prepares the data for training.
        """
        digits = load_digits()
        images = digits['images']
        target = digits['target']
        if sub_data is not None:
            if sub_data == "18":
                images, target = MnistSNN.get_sub_data(images, target, 1, 8)
            elif sub_data == "38":
                images, target = MnistSNN.get_sub_data(images, target, 3, 8)
        one_hot_target = MnistSNN.one_hot_encode_mnist(target)
        normalized_images = MnistSNN.normalize_and_flatten_images(images)
        print(images[0])
        print(normalized_images[0])
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10
        x_train, x_test, y_train, y_test = train_test_split(normalized_images, one_hot_target,
                                                            test_size=1 - train_ratio)

        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=test_ratio / (test_ratio + validation_ratio))

        training_data = (x_train, y_train)
        testing_data = (x_test, y_test)
        validation_data = (x_val, y_val)
        data = (training_data, testing_data, validation_data)

        return data

    @staticmethod
    def get_sub_data(data, target, label1, label2):
        data = data[np.logical_or(target == label1, target == label2)]
        target = target[np.logical_or(target == label1, target == label2)]
        return data, target

    def get_validation_accuracy(self):
        validation_trains = []
        for k in range(len(self.validation_features)):
            input_train = np.array(self.rate_encoding(self.validation_features[k], self.total_time))
            validation_trains.append(input_train)

        decoded_outputs = self.test_step_mnist(validation_trains)
        decoded_test_labels = []
        for tl in self.validation_labels:
            decoded_test_labels.append(tl.index(1))
        return self.get_accuracy(decoded_test_labels, decoded_outputs)

    def execute_mnist(self, execution_name, use_saved_weights=True, visualize_weights=False, validation_gap=100):
        print("\n{} EXECUTION".format(execution_name))
        n_iterations = 1
        input_trains = []
        for i in range(len(self.train_features)):
            input_train = np.array(self.rate_encoding(self.train_features[i], self.total_time))
            input_trains.append(input_train)

        if not use_saved_weights:
            validation_accuracies = []
            for k in range(n_iterations):
                print('Iteration ', k)
                for i in range(len(self.train_features)):
                    if (i + 1) % validation_gap == 0:
                        print("Sample: ", i)
                        validation_accuracies.append(self.get_validation_accuracy())
                    expected_output_train = np.array(
                        self.rate_encoding(self.train_labels[i], self.total_time, delay=1, base_freq=0))
                    self.train_mnist(input_trains[i], expected_output_train)

            np.save("normalized_weights_1.npy", self.weights)
            # self.normalize_weights()
            plt.plot([validation_gap * i for i in range(len(validation_accuracies))], validation_accuracies)
            plt.xlabel("Samples trained on")
            plt.ylabel("Accuracy in percentage")
            plt.show()
        else:
            self.weights = np.load("normalized_weights_1.npy")
        if visualize_weights:
            for i in range(len(self.weights)):
                first_image = np.array(self.weights[i], dtype='float')
                pixels = first_image.reshape((8, 8))
                plt.imshow(pixels, cmap='gray')
                plt.show()
        # self.normalize_weights()
        test_trains = []
        for i in range(len(self.test_features)):
            input_train = np.array(self.rate_encoding(self.test_features[i], self.total_time))
            test_trains.append(input_train)

        decoded_outputs = self.test_step_mnist(test_trains)
        decoded_test_labels = []
        for tl in self.test_labels:
            decoded_test_labels.append(tl.index(1))
        print("Test accuracy is: {}%".format(self.get_accuracy(decoded_test_labels, decoded_outputs)))


if __name__ == '__main__':
    # test_accuracy = 77%
    train_data, test_data, valid_data = MnistSNN.data_extraction(sub_data="18")
    snn = MnistSNN(train_data[0], train_data[1], valid_data[0], valid_data[1], test_data[0], test_data[1])
    snn.execute_mnist("MNIST", use_saved_weights=False, visualize_weights=True, validation_gap=20)

    # test_accuracy = 65%
    train_data, test_data, valid_data = MnistSNN.data_extraction(sub_data="38")
    snn = MnistSNN(train_data[0], train_data[1], valid_data[0], valid_data[1], test_data[0], test_data[1])
    snn.execute_mnist("MNIST", use_saved_weights=False, visualize_weights=True, validation_gap=20)

    train_data, test_data, valid_data = MnistSNN.data_extraction()
    snn = MnistSNN(train_data[0], train_data[1], valid_data[0], valid_data[1], test_data[0], test_data[1])
    snn.execute_mnist("MNIST", use_saved_weights=False, visualize_weights=True, validation_gap=100)
