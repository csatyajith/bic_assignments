import math

import matplotlib.pyplot as plt
import numpy as np

from lif_neuron import LIFNeuron


class SNN:

    def __init__(self):

        self.inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.weights = np.zeros((1, 2))
        self.time = np.arange(0, 1000, 1)
        self.potential_list = []
        for _ in range(4):
            self.potential_list.append([])

        self.A_plus = 0.8
        self.A_minus = 0.3
        self.tau_plus = 8
        self.tau_minus = 5
        self.sliding_window = np.arange(-15 - 1, 15 + 1, 1)
        self.network = []

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

    @staticmethod
    def rate_encoding(inp, time=1000):
        rate_encoded_input = []
        for y in range(len(inp)):
            temp = np.zeros((time,))
            freq = int(inp[y] * 30 + 10)
            spike_gap = math.ceil(1000 / freq)
            i = 0
            while i < 100:
                temp[i] = 1
                i = i + spike_gap
            rate_encoded_input.append(temp)

        return rate_encoded_input

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
                        if 0 <= t + t1 < 100 and t1 != 0:
                            if train[i][t + t1] == 1:
                                # print('weight change by' + str(update_weights(weights[0][i], t1))+ 'for neuron '+str(i))
                                self.weights[0][i] = self.update_weights(self.weights[0][i], t1)

    def execute(self):
        op_neuron = LIFNeuron()

        for k in range(10):
            print('Iteration ', k)
            print('weights', self.weights)
            for i in range(4):
                train = np.array(self.rate_encoding(self.inp[i]))
                op_neuron.reset()
                self.lif(train, i, op_neuron)

        print('Final weights', self.weights)

        for i in range(4):
            total_time = np.arange(0, len(self.potential_list[i]), 1)
            pth = []
            for j in range(len(total_time)):
                pth.append(op_neuron.threshold_v)
            # plotting
            axes1 = plt.gca()
            plt.plot(total_time, pth, 'r')
            plt.plot(total_time, self.potential_list[i])
            plt.show()

        color_codes = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

        plt.eventplot(self.potential_list, color=color_codes)
        plt.title('Spike raster plot')
        plt.xlabel('Spike')
        plt.ylabel('Neuron')
        plt.show()


if __name__ == '__main__':
    spiking = SNN()
    spiking.execute()
