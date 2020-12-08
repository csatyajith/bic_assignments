import matplotlib.pyplot as plt
import numpy as np


class LIFNeuron:
    def __init__(self, resting_v=0, threshold_v=10, membrane_resistance=1, membrane_capacitance=100,
                 v_spike=100):
        """
        Initializes an LIF Neuron with the parameters given.
        :param resting_v: The resting membrane potential of the neuron.
        :param threshold_v: The threshold membrane potential crossing which a spike occurs.
        :param membrane_resistance: The membrane resistance of the neuron.
        :param membrane_capacitance: The membrane capacitance of the neuron.
        :param v_spike: The potential gain associated with a spike.
        """
        self.resting_v = resting_v
        self.threshold_v = threshold_v
        self.membrane_resistance = membrane_resistance
        self.membrane_capacitance = membrane_capacitance
        self.membrane_potential = resting_v
        self.v_spike = v_spike
        self.tau = self.membrane_capacitance * self.membrane_resistance
        self.ref_period = 10
        self.t_inactive = -1

    def reset(self):
        self.t_inactive = -1
        self.membrane_potential = self.resting_v

    def simulate_neuron(self, input_current, time_values, time_delta, report_potentials=False):
        """
        Simulates the potential activity of a neuron.
        :param input_current: The input current to a neuron.
        :param time_values: The time values for which we need to simulate the neuron. This is an array.
        :param time_delta: The time delta which we need to use while computing the membrane potential.
        :return: The list of potentials associated with time values and a count of no. of spikes.
        """
        potentials_list = []
        spike_count = 0
        for _ in time_values:
            self.membrane_potential += time_delta * (
                    (self.membrane_resistance * input_current - self.membrane_potential) / self.tau)
            if self.membrane_potential >= self.threshold_v:
                potentials_list.append(self.membrane_potential + self.v_spike)
                self.membrane_potential = self.resting_v
                spike_count += 1
            else:
                potentials_list.append(self.membrane_potential)
        for i, m in enumerate(potentials_list):
            if i % 100 == 0:
                if report_potentials:
                    print(m)
        return potentials_list, spike_count

    @staticmethod
    def plot_potentials(potentials_list, time_values, input_current):
        plt.plot(time_values, potentials_list)
        plt.title("Membrane_potential for input current value: {}".format(input_current))
        plt.xlabel("Time")
        plt.ylabel("Membrane Potential")
        plt.show()

    @staticmethod
    def plot_spikes_vs_current(spike_counts, currents):
        plt.plot(currents, spike_counts)
        plt.title("Input currents vs spike frequency plot")
        plt.xlabel("Input Current")
        plt.ylabel("Spikes/Unit time")
        plt.show()


def q1(input_current):
    input_current = input_current
    time_delta = 1
    time = 1000
    time_values = list(np.linspace(time_delta, time, int(time / time_delta)))
    new_lif = LIFNeuron()
    potentials_list, spike_count = new_lif.simulate_neuron(input_current, time_values, time_delta)
    new_lif.plot_potentials(potentials_list, time_values, input_current)


def q2():
    input_currents = list(range(1, 200, 1))
    time_delta = 1
    time = 1000
    time_values = list(np.linspace(time_delta, time, int(time / time_delta)))
    spike_counts = []
    for ic in input_currents:
        new_lif = LIFNeuron()
        potentials_list, spike_count = new_lif.simulate_neuron(ic, time_values, time_delta)
        spike_counts.append(spike_count/100)
    LIFNeuron.plot_spikes_vs_current(spike_counts, input_currents)


if __name__ == '__main__':
    q1(input_current=20)
    # q1(input_current=2)
    # q1(input_current=5)
    # q2()
