import sys
import numpy as np
np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import zip_longest
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
import random
import pandas as pd
from numba import jit, njit

SEED = 8
random.seed(SEED)
def shuffle_forward(l):
    order = list(range(len(l)))
    random.shuffle(order)
    return np.array(np.array(l)[order]), order

def shuffle_backward(l, order):
    l_out = [0] * len(l)
    for i, j in enumerate(order):
        l_out[j] = l[i]
    return l_out


@njit
def nb_LIF(I, dT, V_rest, V_thresh, V_spike, Rm, Cm):
    """
    Runs a LIF simulation on neuron and returns outputted voltage

            Parameters:
                    I (double[]): A numpy array of input voltages in mV

            Returns:
                    V (double[]): A numpy array of the output voltages in mV
    """
    total_time = I.size * dT

    # an array of time
    time = np.arange(0, total_time, dT)

    # default voltage list set to resting volatage of -65mV
    V = V_rest * np.ones(len(time))

    did_spike = False

    for t in range(len(time)):
        # using "I - V(t)/Rm = Cm * dV/dT"
        dV = (I[t] - (V[t - 1] - V_rest) / Rm) / Cm

        # reset membrane potential if neuron spiked last tick
        if did_spike:
            V[t] = V_rest + dV * dT
        else:
            V[t] = V[t - 1] + dV * dT

        # check if membrane voltage exceeded threshold (spike)
        if V[t] > V_thresh:
            did_spike = True
            # set the last step to spike value
            V[t] = V_spike
        else:
            did_spike = False

    return V


@njit
def nb_voltage_to_output(V_input, V_spike):
    """Converts a neuron's internal voltage to output"""
    V_output = np.zeros(V_input.shape)
    for i, v in enumerate(V_input):
        V_output[i] = 0 if v < V_spike else V_spike
    return V_output


@njit
def nb_voltage_to_spike_rate(voltages, V_spike, dT, rate):
    """Converts an array of neuron voltages to spikes per n seconds, where n = rate"""
    #         print('voltages', voltages)

    spike_count = 0
    for v in voltages:
        if v >= V_spike:
            spike_count += 1

    #         print('spike_count', spike_count)

    total_time_dT = len(voltages) * dT
    #         print(f'total_time_dT: {total_time_dT} ({dT}ms)')

    spikes_per_dT = spike_count / total_time_dT
    #         print(f'spikes_per_ms: {spikes_per_dT} (spikes/ms)')

    return spikes_per_dT * 1000 * rate


@njit
def nb_new_weight(weight, a_corr, input_rate, output_rate, w_max, w_decay):
    # adjust the weight using Hebb with decay
    weight_change = a_corr * input_rate * output_rate - w_decay
    #                         print('\told weight', weight)
    #                         print('\tweight_change:', weight_change)
    new_weight = 0
    if weight + weight_change < 0:
        new_weight = 0
    elif weight + weight_change > w_max:
        new_weight = w_max
    else:
        new_weight = weight + weight_change
    return new_weight


class SingleLayerSNN:
    def __init__(
            self,
            inputs,
            weights,
            trainings,
            Cm=4.0,
            Rm=5.0,
            V_thresh=30.0,
            V_rest=-65.0,
            V_spike=80.0,
            dT=0.01,
            rate=1.0,
    ):
        """
        Runs a LIF simulation on neuron and returns outputted voltage

                Parameters:
                        inputs (double[][][]): A 3d numpy array of the input voltages per timestep
                        weights (double[]): A numpy array of initial weights
                        outputs (double[][][]): A 3d numpy array of the output voltages per timestep used for teaching neuron
                Returns:
                        None
        """

        assert len(trainings) == len(inputs)

        self.inputs = inputs
        self.weights = weights
        self.trainings = trainings
        self.Cm = Cm
        self.Rm = Rm
        self.V_thresh = V_thresh
        self.V_rest = V_rest
        self.V_spike = V_spike
        self.dT = dT  # ms
        self.rate = rate  # sec
        self._LIF_spikes = 0

    def LIF(self, I):
        """
        Runs a LIF simulation on neuron and returns outputted voltage

                Parameters:
                        I (double[]): A numpy array of input voltages in mV

                Returns:
                        V (double[]): A numpy array of the output voltages in mV
        """
        return nb_LIF(I, self.dT, self.V_rest, self.V_thresh, self.V_spike, self.Rm, self.Cm)

    def voltage_to_output(self, V_input):
        """Converts a neuron's internal voltage to output"""
        return nb_voltage_to_output(V_input, self.V_spike)

    def voltage_to_spike_rate(self, voltages, dT=None, rate=None):
        """Converts an array of neuron voltages to spikes per n seconds, where n = rate"""
        if not dT:
            dT = self.dT
        if not rate:
            rate = self.rate

        return nb_voltage_to_spike_rate(voltages, self.V_spike, dT, rate)

    # returns the voltages of input and output neurons
    def feed_forward(self, inputs, train=True):
        """
        Passes all sets of inputs to

                Parameters:
                        inputs (double[][][]): A 3d numpy array that contains every set of inputs voltages for each input neuron
                        train (boolean): Determines whether or not to inject training voltages

                Returns:
                        all_input_voltages, all_output_voltages (double[][][], double[][][]): A tuple of all of the input voltages and all of the output voltages
        """
        all_input_voltages = np.zeros(inputs.shape)
        all_output_voltages = np.zeros(self.trainings.shape)

        # set the training voltages to all zero if not running in training mode
        training_copy = (
            np.zeros(inputs.shape, np.ndarray)
            if not train
            else np.array(self.trainings)
        )
        #         print('training_copy:')
        #         print(training_copy)

        inputs_copy = np.array(inputs)
        #         print('inputs_copy:')
        #         print(inputs_copy)

        assert len(training_copy) == len(inputs_copy)

        zipped = list(zip(inputs_copy, training_copy))

        shuffled, order = shuffle_forward(zipped)
        shuffled_inputs, shuffled_trainings = zip(*shuffled)

        shuffled_inputs = np.array(shuffled_inputs)
        #         print('shuffled_inputs:')
        #         print(shuffled_inputs)

        shuffled_trainings = np.array(shuffled_trainings)
        #         print('shuffled_trainings:')
        #         print(shuffled_trainings)

        # feed inputs through input neurons to get weighted voltage for output neurons
        for i, (input_set, training_set) in enumerate(zip_longest(shuffled_inputs, shuffled_trainings)):

            for j, V_input in enumerate(input_set):
                temp = np.array(V_input)
                all_input_voltages[i][j] = self.LIF(temp.astype(float))
            input_voltages = all_input_voltages[i]

            output_inputs = np.zeros(self.trainings[0].shape)
            #             input_outputs = []  # DEBUG ONLY
            for j, weight_set in enumerate(self.weights.T):
                weighted_sum = np.zeros(len(input_set[0]))
                for V_input, weight in zip(input_voltages, weight_set):
                    # filter for spikes b/c a neuron only outputs if it spikes
                    input_output = self.voltage_to_output(V_input)
                    #                     input_outputs.append(input_output)  # DEBUG ONLY
                    weighted = input_output * weight
                    weighted_sum = np.add(weighted_sum, weighted)

                output_inputs[j] = weighted_sum

            #             input_outputs = np.array(input_outputs)

            #             print('input_voltages:')
            #             print(input_voltages)
            #             print('input_outputs:')
            #             print(input_outputs)
            #             print('output_inputs:')
            #             print(output_inputs)
            #             print('training_set:')
            #             print(training_set)

            # inject training voltage if in training mode
            assert isinstance(training_set, (list, np.ndarray))
            for j, (output_input, training_input) in enumerate(
                    zip(output_inputs, training_set)
            ):
                assert isinstance(training_input, (list, np.ndarray))
                #               padded_training_input = np.pad(training_input, (0, len(output_inputs) - len(training_set)), "constant")
                output_inputs[j] = output_input + training_input

            #             print('output_inputs after injecting training current')
            #             print(output_inputs)

            # run LIF on output neurons
            for j, V_input in enumerate(output_inputs):
                all_output_voltages[i][j] = self.LIF(V_input)

            output_voltages = all_output_voltages[i]

        #             print('output_voltages:')
        #             print(output_voltages)

        # unshuffle the voltages
        all_input_voltages = np.array(shuffle_backward(all_input_voltages, order))
        all_output_voltages = np.array(shuffle_backward(all_output_voltages, order))

        return all_input_voltages, all_output_voltages

    def train(self, epochs=75, a_corr=0.000000002, w_max=2, w_decay=0, show_legend=True):
        """Runs feed forward with training dataset"""
        weights_history = []
        for weight_set in self.weights:
            weight_row = []
            for weight in weight_set:
                weight_row.append([weight])
            weights_history.append(weight_row)

        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")

            print('\trunning feed forward...')
            all_input_voltages, all_output_voltages = self.feed_forward(
                self.inputs, train=True
            )

            # debug info
            #             print()
            #             print('------------------------------------------------')
            #             print('all_input_voltages:')
            #             print(all_input_voltages)
            #             print('all_output_voltages:')
            #             print(all_output_voltages)

            #             print('weights:')
            #             print(self.weights)

            print('\tapplying learning rule...')
            # apply learning rule
            for input_voltages, output_voltages in zip(
                    all_input_voltages, all_output_voltages
            ):
                #                 print('input_voltages:')
                #                 print(input_voltages)
                for i, (input_voltage_set, weight_set) in enumerate(
                        zip(input_voltages, self.weights)
                ):
                    #                     print('input_voltage_set', input_voltage_set)
                    input_rate = self.voltage_to_spike_rate(input_voltage_set)
                    #                     print(f'input_rate {i}:', input_rate)

                    for j, (output_voltage_set, weight) in enumerate(
                            zip(output_voltages, weight_set)
                    ):
                        output_rate = self.voltage_to_spike_rate(output_voltage_set)
                        #                         print(f'\toutput_rate {j}:', output_rate)

                        # adjust the weight using Hebb with decay
                        self.weights[i][j] = nb_new_weight(self.weights[i][j], a_corr, input_rate, output_rate, w_max,
                                                           w_decay)

            #                         print('\tnew weight', self.weights[i][j], '\n')

            # update weight history
            for i, weight_set in enumerate(self.weights):
                for j, weight in enumerate(weight_set):
                    weights_history[i][j].append(weight)

            print('\tweights:')
            print('\t' + str(self.weights).replace('\n', '\n\t'))

        # plot neuron spiking data
        #             for i, (input_voltages, output_voltages) in enumerate(zip(all_input_voltages, all_output_voltages)):
        #                 plt.figure(figsize=(20,10))
        #                 plt.title(f'Input: {i + 1}')
        #                 for input_voltage in input_voltages:
        #                     plt.plot(input_voltage, 'b:', alpha=.5)

        #                 for output_voltage in output_voltages:
        #                     plt.plot(output_voltage, 'r--', alpha=.5)

        #             plt.show()

        # plot weights history
        plt.figure(figsize=(20, 10))
        for i, weight_row in enumerate(weights_history):
            for j, weight_history in enumerate(weight_row):
                plt.plot(weight_history, label=f"weight {i}-{j}")
                plt.xlabel('Epoch')
        if show_legend:
            plt.legend(loc="upper left")
        plt.title("Weight History")
        plt.show()

    def _generate_bar_plot(self, outputs, title='', ylabel='', stacked=False):
        d = {}
        """Generates bar plot for output neuron data"""
        for x, bars in enumerate(outputs.T):
            d[f'Output Neuron {x}'] = bars
        index = [f'{x}' for x in range(len(outputs.T[0]))]
        df = pd.DataFrame(d, index=index)
        ax = df.plot.bar(rot=0, figsize=(20, 10), stacked=stacked, title=title)
        ax.set_xlabel('Input Set')
        ax.set_ylabel(ylabel)

    def predict(self, inputs,bar_width=0.25):
        """Runs feed foward without training data on inputs"""
        all_input_voltages, all_output_voltages = self.feed_forward(inputs, train=False)
        #         print('all_output_voltages')
        #         print(all_output_voltages)

        all_output_confidences = []
        all_output_spike_rates = []

        for x, (input_voltages, output_voltages) in enumerate(
                zip(all_input_voltages, all_output_voltages)
        ):
            output_spike_rates = []
            output_confidences = []
            print("input set:", x)
            for i, input_voltage_set in enumerate(input_voltages):
                print(
                    f"\tinput {i}: {self.voltage_to_spike_rate(input_voltage_set)} spikes/{self.rate}s"
                )
            print()

            output_voltages_sum = sum(
                [self.voltage_to_spike_rate(v) for v in output_voltages]
            )
            for i, output_voltage_set in enumerate(output_voltages):
                sr = self.voltage_to_spike_rate(output_voltage_set)
                output_spike_rates.append(sr)
                confidence = (
                    "N/A" if output_voltages_sum == 0 else sr / output_voltages_sum
                )
                output_confidences.append(0 if confidence == "N/A" else confidence)
                print(
                    f"\toutput {i}: {sr} spikes/{self.rate}s, confidence: {confidence}"
                )
            print()
            all_output_spike_rates.append(output_spike_rates)
            all_output_confidences.append(output_confidences)
            # prediction = np.argmax(output_confidences)
            #
            # print("predicted value: ", prediction)
            # val = target[x]
            # if (prediction == val):
            #     x1= True
            # else:
            #     x1= False
            # correct=0
            # if x1:
            #     correct += 1
            # accuracy =  correct / len(target)



        # print("acuuracy:",accuracy)
        all_output_confidences = np.array(all_output_confidences)
        all_output_spike_rates = np.array(all_output_spike_rates)



        #         print(all_output_confidences)
        #         print(all_output_spike_rates)

        # plot output confidence
        self._generate_bar_plot(all_output_confidences, title="Output Confidence", ylabel="Confidence", stacked=True, )
        # plot output spike rates
        self._generate_bar_plot(
            all_output_spike_rates,
            ylabel=f"Spike Rate (spikes/{self.rate}s)",
            title="Output Spike Rates",

        )
        return all_input_voltages, all_output_voltages


if __name__ == '__main__':
    digits = load_digits()

    u_t = 2000  # units of time
    t_a = 1000  # training injection amount
    s_a = -100  # training supression amount

    input_limit = 100
    pixels_per_digit = len(digits.data[0])
    intensity_multiplier = 500
    digit_types = 10  # max: 10, min: 1

    # preprocess data
    inputs = np.zeros((input_limit, pixels_per_digit, u_t), float)
    trainings = np.zeros((input_limit, digit_types, u_t), float)

    for i, (digit, target) in enumerate(zip(digits.data, digits.target[:input_limit])):
        digit_voltage_stream = np.zeros((pixels_per_digit, u_t), float)
        training_voltage_stream = np.zeros((digit_types, u_t), float)

        for j, pixel in enumerate(digit):
            digit_voltage_stream[j] = np.array([pixel * intensity_multiplier] * u_t)

        assert target < digit_types
        for j in range(digit_types):
            training_voltage_stream[j] = np.array([t_a] * u_t) if j == target else np.array([s_a] * u_t)

        inputs[i] = digit_voltage_stream
        trainings[i] = training_voltage_stream

    weights = np.zeros((pixels_per_digit, digit_types))

    # feed into network
    digit_network = SingleLayerSNN(inputs=inputs[:input_limit], weights=weights, trainings=trainings)
    digit_network.train(epochs=28, a_corr=0.0000000001, w_max=2.0, w_decay=0.0001, show_legend=False)

    # predict the first 10 images in the dataset
    # digit_network.predict(inputs[:10],digits.target[:10],bar_width=0.1);
    digit_network.predict(inputs[:10], bar_width=0.1);


    # predict second 10 images
    digit_network.predict(inputs[10:20], bar_width=0.1);

    # predict 100th 10 images (new data)
    digit_network.predict(inputs[20:30], bar_width=0.1);

    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2])
    b = np.pad(b, (0, len(a) - len(b)), "constant")
    c = [2, 3, 5]

    shuffled, order = shuffle_forward(a)
    print('shuffled:', shuffled)
    unshuffled = shuffle_backward(shuffled, order)
    print('unshuffled:', unshuffled)

    np.repeat([[1], [2], [3]], 4, axis=1)



