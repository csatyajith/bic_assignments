import matplotlib.pyplot as plt
import numpy as np


class LIFNeuron:
    def __init__(self, resting_v=-65, threshold_v=30, membrane_resistance=1, membrane_capacitance=10,
                 v_spike=100):
        self.resting_v = resting_v
        self.threshold_v = threshold_v
        self.membrane_resistance = membrane_resistance
        self.membrane_capacitance = membrane_capacitance
        self.membrane_potential = resting_v
        self.v_spike = v_spike
        self.tau = self.membrane_capacitance * self.membrane_resistance

    def simulate_neuron(self, input_current, time_values, time_delta):
        potentials_list = []
        for _ in time_values:
            self.membrane_potential = self.membrane_potential + time_delta * (
                    (self.membrane_resistance * input_current - self.membrane_potential) / self.tau)
            if self.membrane_potential >= self.threshold_v:
                potentials_list.append(self.membrane_potential + self.v_spike)
                self.membrane_potential = self.resting_v
            else:
                potentials_list.append(self.membrane_potential)
        return potentials_list

    def plot_potentials(self, potentials_list, time_values):
        plt.plot(time_values, potentials_list)
        plt.xlabel("Time")
        plt.ylabel("Membrane Potential")
        plt.show()


def q1():
    input_current = 10000
    time_delta = 0.02
    time_values = list(np.linspace(time_delta, 1, int(1 / time_delta)))
    new_lif = LIFNeuron()
    potentials_list = new_lif.simulate_neuron(input_current, time_values, time_delta)
    new_lif.plot_potentials(potentials_list, time_values)


if __name__ == '__main__':
    q1()
