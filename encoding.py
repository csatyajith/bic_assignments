import math

import numpy as np

from lif_neuron import LIFNeuron


class SNN:

    def __init__(self):

        self.inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.outputs = np.array([0, 0, 0, 1])
        self.weights = np.zeros((1, 2))
        self.total_time = 1000
        self.time = np.arange(0, self.total_time, 1)
        self.potential_list = []
        for _ in range(4):
            self.potential_list.append([])

        self.A_plus = 0.8
        self.A_minus = 0.3
        self.tau_plus = 8
        self.tau_minus = 5
        self.sliding_window = np.arange(-10 - 1, 10 + 1, 1)
        self.network = []
        self.weight_init()
        self.op_neuron = LIFNeuron()

    def weight_init(self):
        for i in range(1):
            for j in range(2):
                self.weights[i][j] = np.random.uniform(0, 1)

    def create_network(self, n_input_neurons, n_output_neurons):
        self.network.append([LIFNeuron() for _ in range(n_input_neurons)])
        self.network.append([LIFNeuron() for _ in range(n_input_neurons)])
        for i in n_output_neurons:
            for j in n_input_neurons:
                self.weights[i][j] = np.random.uniform(0, 1)

    def update_weights(self, w, t):
        if t > 0:
            delta_w = -self.A_plus * np.exp(-float(t) / self.tau_plus)
        else:
            delta_w = self.A_minus * np.exp(float(t) / self.tau_minus)

        if delta_w < 0:
            return w + 0.5 * delta_w * w
        elif delta_w > 0:
            return w + 0.5 * delta_w * (1 - w)

    def rate_encoding(self, inp, time=100):
        spike_train = []
        for y in range(2):
            spike_train.append(self.rate_encoding_output(inp[y], time))

        return spike_train

    @staticmethod
    def rate_encoding_output(op, time):
        spike_train = np.zeros((time,))
        freq = (op * 80 + 10)
        freq = math.ceil(1000 / freq)
        i = 0
        while i < time:
            spike_train[i] = 1
            i = i + freq

        return spike_train

    @staticmethod
    def spikes_to_current(a_spikes, conversion_factor):
        return (sum(a_spikes) / len(a_spikes)) * conversion_factor * 6000

    @staticmethod
    def decode_output(n_spikes_list, positive_threshold):
        outputs = []
        for n in n_spikes_list:
            if n > positive_threshold:
                outputs.append(1)
            else:
                outputs.append(0)
        return outputs

    def test_step(self, input_trains):
        output_spike_counts = []
        for inp in input_trains:
            self.op_neuron.reset()
            current = 0
            for i in range(len(inp)):
                current += self.spikes_to_current(inp[i], self.weights[0][i])
            time_values = list(np.linspace(1, self.total_time, self.total_time))
            potentials_list, spike_count = self.op_neuron.simulate_neuron(current, time_values, 1)
            output_spike_counts.append(spike_count)
        return output_spike_counts

    def train_step(self, input_train, output_train):
        for t in self.time:
            if output_train[t] == 1:
                for i in range(2):
                    for t1 in self.sliding_window:
                        if 0 <= t + t1 < self.total_time and t1 != 0 and input_train[i][t + t1] == 1:
                            self.weights[0][i] = self.update_weights(self.weights[0][i], t1)

    def lif(self, train, n, op_neuron):
        for t in self.time:
            if op_neuron.t_inactive < t:
                op_neuron.membrane_potential += np.dot(self.weights[0], train[:, t])

            self.potential_list[n].append(op_neuron.membrane_potential)
            print('membrane pot', op_neuron.membrane_potential)
            if op_neuron.membrane_potential >= op_neuron.threshold_v:
                op_neuron.t_inactive = t + op_neuron.ref_period
                print('time inactive at t =', op_neuron.t_inactive, t)
                op_neuron.membrane_potential = op_neuron.resting_v
                for i in range(2):
                    for t1 in self.sliding_window:
                        if 0 <= t + t1 < self.total_time and t1 != 0:
                            if train[i][t + t1] == 1:
                                self.weights[0][i] = self.update_weights(self.weights[0][i], t1)

    def get_accuracy(self, expected_output, actual_output):
        success = 0
        for i in range(len(expected_output)):
            if expected_output[i] == actual_output[i]:
                success += 1
        return (success / len(expected_output)) * 100

    def execute(self, execution_name):
        print("\n{} EXECUTION".format(execution_name))
        n_iterations = 10
        input_trains = []
        for i in range(len(self.inp)):
            input_train = np.array(self.rate_encoding(self.inp[i], self.total_time))
            input_trains.append(input_train)

        for k in range(n_iterations):
            print('Iteration ', k, '; weights: ', self.weights)
            for i in range(len(self.inp)):
                expected_output_train = np.array(self.rate_encoding_output(self.outputs[i], self.total_time))
                self.train_step(input_trains[i], expected_output_train)

        print('Final weights', self.weights)
        encoded_outputs = self.test_step(input_trains)
        decoded_outputs = self.decode_output(encoded_outputs, 90)
        print("Inputs are: ", self.inp, " Expected outputs are: ", self.outputs)
        print("Number of output spikes by input: ", encoded_outputs)
        print("Decoded output values from the spikes: ", decoded_outputs)
        print("Accuracy is: {}%".format(self.get_accuracy(self.outputs, decoded_outputs)))

    def execute_and(self):
        self.execute("AND")
        # for i in range(len(self.inp)):
        #     total_time = np.arange(0, len(self.potential_list[i]), 1)
        #     pth = []
        #     for j in range(len(total_time)):
        #         pth.append(op_neuron.threshold_v)
        #     # plotting
        #     axes1 = plt.gca()
        #     plt.plot(total_time, pth, 'r')
        #     plt.plot(total_time, self.potential_list[i])
        #     plt.show()
        #
        # color_codes = np.array([[0, 0, 0],
        #                         [1, 0, 0],
        #                         [0, 1, 0],
        #                         [0, 0, 1]])
        #
        # plt.eventplot(self.potential_list, color=color_codes)
        # plt.title('Spike raster plot')
        # plt.xlabel('Spike')
        # plt.ylabel('Neuron')
        # plt.show()

    def execute_or(self):
        self.outputs = [0, 1, 1, 1]
        self.execute("OR")


if __name__ == '__main__':
    spiking = SNN()
    spiking.execute_and()
    spiking.execute_or()
