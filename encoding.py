import math

import matplotlib.pyplot as plt
import numpy as np

from lif_neuron import LIFNeuron


class SNN:

    def __init__(self):

        self.inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        self.time = np.arange(0, 100, 1)
        self.A_plus = 0.8
        self.A_minus = 0.8
        self.tau_plus = 8
        self.tau_minus = 5
        self.sliding_window = np.arange(-20-1, 20+1, 1)
        self.network = []
        self.layers = 2
        self.no_of_neurons = [2, 1]
        self.weights = []
        self.potential_list = []  # output neuron potential values
        self.input_potential = []
        for _ in range(self.no_of_neurons[self.layers - 1]):
            self.potential_list.append([])
        for _ in range(4):  # potential list for each input
            self.input_potential.append([])

    def reset(self, layer):
        for n in self.network[layer]:
            n.t_inactive = -1
            n.membrane_potential = n.resting_v

    def create_network(self):
        # self.network.append([LIFNeuron() for _ in range(n_input_neurons)])
        # self.network.append([LIFNeuron() for _ in range(n_input_neurons)])

        for l in range(0, self.layers):
            neurons = [LIFNeuron() for _ in range(self.no_of_neurons[l])]
            self.network.append(neurons)
        print(self.network)

        for i in range(0, self.layers - 1):
            local_weights = np.random.uniform(0, 1, (self.no_of_neurons[i + 1], self.no_of_neurons[i]))
            self.weights.append(local_weights)

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
    def rate_encoding(inp, time=100):
        rate_encoded_input = []
        for y in range(len(inp)):
            temp = np.zeros((time,))
            freq = int(inp[y] * 30 + 10)
            spike_gap = math.ceil(100 / freq)
            i = spike_gap
            while i < 100:
                temp[i] = 1
                i = i + spike_gap
            rate_encoded_input.append(temp)

        return rate_encoded_input

    def train(self, input_train, ip, thresh_v):
        train = input_train
        for l in range(self.layers - 1):
            # print('Layer', l)
            self.reset(l + 1)
            spike_train_layer = []

            for n in range(self.no_of_neurons[l + 1]):
                spike_train_layer.append([])

            for t in self.time:
                # print(len(self.network[l + 1]))
                print('Train at t', train[:, t])
                for j, n in enumerate(self.network[l + 1]):
                    if t > n.t_inactive:
                        # print('weight', self.weights[l][j], l, j)
                        n.membrane_potential += np.dot(self.weights[l][j], train[:, t])
                        if n.membrane_potential > n.resting_v:
                            n.membrane_potential -= 0.11

                    self.potential_list[j].append(n.membrane_potential)
                    self.input_potential[ip].append(n.membrane_potential)
                for j, n in enumerate(self.network[l + 1]):
                    print('membrane pot', n.membrane_potential)
                    if n.membrane_potential >= thresh_v:
                        # print('threshold')
                        n.t_inactive = t + n.ref_period
                        n.membrane_potential = n.resting_v
                        spike_train_layer[j].append(1)
                        # print('time inactive at t =', n.t_inactive, t)
                        for i in range(self.no_of_neurons[l]):
                            for t1 in self.sliding_window:
                                if 0 <= t + t1 < 100 and t1 != 0:
                                    if train[i][t + t1] == 1:
                                        print('t1', t1)
                                        v = self.update_weights(self.weights[l][j][i], t1)
                                        print('weight change by' + str(v) + 'for neuron ' + str(i))
                                        self.weights[l][j][i] = v
                    else:
                        spike_train_layer[j].append(0)

            print('spike train for layer', spike_train_layer)
            train = np.array(np.asarray(spike_train_layer))

        return train

    def execute(self, th, spike_th):

        for k in range(10):
            print('Iteration ', k)
            print('weights', self.weights)
            for i in range(4):
                spike_train = np.array(self.rate_encoding(self.inp[i]))
                # print('spike train', spike_train)
                op_train = self.train(spike_train, i, th)
                n_spikes = sum(op_train[0])
                print('Actual', i, 'Spike frequency', n_spikes)
                out = 1 if n_spikes >= spike_th else 0
                print('Output', out)
                input_current = n_spikes / len(op_train[0])
                print('net', self.network)
                self.network[1][0].simulate_neuron(input_current, np.arange(len(op_train)), 1)

        print('Final weights', self.weights)

        for i in range(self.no_of_neurons[self.layers - 1]):
            total_time = np.arange(0, len(self.potential_list[i]), 1)
            pth = []
            for j in range(len(total_time)):
                pth.append(th)

            plt.plot(total_time, pth, 'r')
            plt.plot(total_time, self.potential_list[i])
            plt.show()

        color_codes = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])

        plt.eventplot(self.input_potential, color=color_codes)
        plt.title('Spike raster plot')
        plt.xlabel('Spike')
        plt.ylabel('Neuron')
        plt.show()


if __name__ == '__main__':
    spiking = SNN()
    spiking.create_network()
    thresh = -64
    spike_thresh = 4

    # if type == 'OR':
    # thresh = -64.3
    # spike_thresh = 1

    spiking.execute(thresh, spike_thresh)
