import math

import matplotlib.pyplot as plt
import numpy as np

from lif_neuron import LIFNeuron


class BinarySNN:

    def __init__(self):

        self.inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.n_outputs = 1
        self.n_labels = 1
        self.input_length = len(self.inp[0])
        self.outputs = np.array([0, 0, 0, 1])
        self.weights = np.zeros((self.n_outputs, self.input_length))
        self.total_time = 1000
        self.time = np.arange(0, self.total_time, 1)

        self.A_plus = 0.8
        self.A_minus = 0.3
        self.tau_plus = 8
        self.tau_minus = 5
        self.sliding_window = np.arange(-10 - 1, 10 + 1, 1)
        self.network = []
        self.weight_init()
        self.op_neuron = LIFNeuron()

    def weight_init(self):
        for i in range(self.n_labels):
            for j in range(self.input_length):
                self.weights[i][j] = np.random.uniform(0, 1)

    def update_weights(self, w, t):
        """
        Uses the weight update rule for STDP. Referred to the following paper: Implementation of Boolean AND and OR
        Logic Gates with Biologically Reasonable Time Constants in Spiking Neural Networks - Lakshay Sahni et al.
        :param w: prior weights
        :param t: time difference between the pre-synaptic spike and the post-synaptic spike
        :return:
        """
        if t > 0:
            delta_w = -self.A_plus * np.exp(-float(t) / self.tau_plus)
        else:
            delta_w = self.A_minus * np.exp(float(t) / self.tau_minus)

        if delta_w < 0:
            return w + 0.5 * delta_w * w
        elif delta_w > 0:
            return w + 0.5 * delta_w * (1 - w)

    def rate_encoding(self, binary_list, time=1000, delay=0, base_freq=10):
        """
        Gets a rate encoded list of spike trains for a list of binary values.
        :param binary_list: The list of binary values for which spike trains need to be generated
        :param time: The time in milliseconds for which each spike train should be generated
        :param delay: The delay with which the first spike should start
        :param base_freq: The base frequency in case the input value is 0
        :return: A list of spike trains each corresponding to an input
        """
        spike_train = []
        for y in binary_list:
            spike_train.append(self.get_spike_train(y, time, delay, base_freq))

        return spike_train

    @staticmethod
    def get_spike_train(binary_inp, time, delay=0, base_freq=10):
        """
        Gets a rate encoded list of spike trains for a single binary input
        :param binary_inp: The binary value for which a spike train needs to be generated
        :param time: The time in milliseconds for which the spike train should be generated
        :param delay: The delay with which the first spike should start
        :param base_freq: The base frequency in case the input value is 0
        :return: A spike train each corresponding to an input
        """
        spike_train = np.zeros((time,))
        freq = (binary_inp * 80 + base_freq)
        if freq == 0:
            return spike_train
        freq = math.ceil(1000 / freq)
        i = freq + delay
        while i < time:
            spike_train[i] = 1
            i = i + freq

        return spike_train

    @staticmethod
    def spikes_to_current(spike_train, weights):
        """
        Converts spike train to current using weights
        :param spike_train: Spike train for which current needs to be generated
        :param weights: The weights associated with that synapse
        :return: The current associated with the spike train and the weights
        """
        return (sum(spike_train) / len(spike_train)) * weights * 6000

    @staticmethod
    def decode_output(n_spikes_list, positive_threshold):
        """
        Gets the predicted binary output from the spikes list of the neuron
        :param n_spikes_list: The number of spikes for each input organized in a list
        :param positive_threshold: The number of spikes exceeding which a positive output is decoded
        :return: The binary outputs corresponding to the spikes list
        """
        outputs = []
        for n in n_spikes_list:
            if n > positive_threshold:
                outputs.append(1)
            else:
                outputs.append(0)
        return outputs

    def train_step(self, input_trains, output_train):
        """
        Trains the weights of the network using the input spike trains and the corresponding expected output spike
        trains. The weights are stored as class variables so this function can be called independently for each sample.
        :param input_trains: The input trains representing a sample
        :param output_train: The expected output train corresponding to the input sample.
        :return:
        """
        for t in self.time:
            if output_train[t] == 1:
                for i in range(2):
                    for t1 in self.sliding_window:
                        if 0 <= t + t1 < self.total_time and t1 != 0 and input_trains[i][t + t1] == 1:
                            self.weights[0][i] = self.update_weights(self.weights[0][i], t1)

    def test_step(self, input_trains):
        """
        For a set of input spike trains, this function predicts the output
        :param input_trains: The input spike trains
        :return: The number of spikes generated by the output neuron in one second for the above input
        """
        output_spike_counts = []
        potentials_lists = []
        for inp in input_trains:
            self.op_neuron.reset()
            current = 0
            for i in range(len(inp)):
                current += self.spikes_to_current(inp[i], self.weights[0][i])
            time_values = list(np.linspace(1, self.total_time, self.total_time))
            potentials_list, spike_count = self.op_neuron.simulate_neuron(current, time_values, 1)
            potentials_lists.append(potentials_list)
            output_spike_counts.append(spike_count)
        return output_spike_counts, potentials_lists

    @staticmethod
    def get_accuracy(expected_output, actual_output):
        """
        Gives the percentage of predictions that were correct
        :param expected_output: The predictions
        :param actual_output: The actual output
        :return: Percentage of predictions that match the output
        """
        success = 0
        for i in range(len(expected_output)):
            if expected_output[i] == actual_output[i]:
                success += 1
        return (success / len(expected_output)) * 100

    def execute(self, execution_name):
        print("\nLOGICAL {} EXECUTION".format(execution_name))
        n_iterations = 10
        input_trains = []
        for i in range(len(self.inp)):
            input_train = np.array(self.rate_encoding(self.inp[i], self.total_time))
            input_trains.append(input_train)

        for k in range(n_iterations):
            print('Iteration ', k, '; weights: ', self.weights)
            for i in range(len(self.inp)):
                expected_output_train = np.array(self.get_spike_train(self.outputs[i], self.total_time))
                self.train_step(input_trains[i], expected_output_train)

        print('Final weights', self.weights)
        encoded_outputs, potentials_list = self.test_step(input_trains)
        decoded_outputs = self.decode_output(encoded_outputs, 90)
        print("Inputs are: ", self.inp, " Expected outputs are: ", self.outputs)
        print("Number of output spikes by input: ", encoded_outputs)
        print("Decoded output values from the spikes: ", decoded_outputs)
        print("Accuracy is: {}%".format(self.get_accuracy(self.outputs, decoded_outputs)))

        raster_plot_outputs = []
        for pl in potentials_list:
            r = []
            for i in range(100):
                if pl[i] >= self.op_neuron.v_spike:
                    r.append(i)
            raster_plot_outputs.append(r)

        color_codes = np.array([[0, 1, 1],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

        plt.eventplot(raster_plot_outputs, color=color_codes)
        plt.title('Spike raster plot for logical {}'.format(execution_name))
        plt.xlabel('Spike')
        plt.ylabel('Neuron')
        plt.legend(("[0,0]", "[0,1]", "[1,0]", "[1,1]"), ncol=4)
        plt.show()

    def execute_and(self):
        self.execute("AND")

    def execute_or(self):
        self.outputs = [0, 1, 1, 1]
        self.weight_init()
        self.execute("OR")

    def execute_xor(self):
        self.outputs = [0, 1, 1, 0]
        self.weight_init()
        self.execute("XOR")


if __name__ == '__main__':
    spiking = BinarySNN()
    spiking.execute_and()
    spiking.execute_or()
    spiking.execute_xor()
