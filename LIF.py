#  LIF neuron
import numpy as np
import matplotlib.pyplot as plt


class LIFModel:
    def __init__(self, inputCurrent):
        self.inputCurrent = inputCurrent

        # all neuron properties
        self.time = 100
        self.delta_time = 0.01
        self.time_values = np.arange(0, self.time, self.delta_time)
        self.m_potential = np.zeros(len(self.time_values))    # Membrane potential
        self.membrane_Resistance = 1  # Resistance
        self.membrane_Capacitance = 10  # Capacitance
        self.tau = self.membrane_Resistance * self.membrane_Capacitance     # tau
        self.V_threshold = 1  # Threshold voltage
        self.V_spike = 0.5  # Spike delta

    def plotNeuronSimulation(self):
        for i in range(1, len(self.time_values)):
            #  approximate v_m using Euler's method
            self.m_potential[i] = self.m_potential[i - 1] + self.delta_time * ((-self.m_potential[i - 1] + self.membrane_Resistance * self.inputCurrent) / self.tau)
            # membrane potential reaches constant threshold
            if self.m_potential[i] >= self.V_threshold:
                # spike occurs
                self.m_potential[i - 1] = self.m_potential[i - 1] + self.V_spike
                # voltage is setback to resting potential
                self.m_potential[i] = 0
        plt.plot(self.time_values, self.m_potential)
        plt.xlabel("Time")
        plt.ylabel("Membrane Potential in volts")
        plt.title("Membrane Potential of a Neuron over Time with Input Current =" + str(self.inputCurrent))
        plt.show()

    def plotNeuronSpikes(self):
        spikes = []
        inputCurrent = np.arange(1, 5, 0.05)
        for current in inputCurrent:
            numSpikes = 0
            for i in range(1, len(self.time_values)):
                self.m_potential[i] = self.m_potential[i - 1] + self.delta_time * ((-self.m_potential[i - 1] + self.membrane_Resistance * current) / self.tau)
                # membrane potential reaches constant threshold
                if self.m_potential[i] >= self.V_threshold:
                    # spike occurs
                    self.m_potential[i - 1] = self.m_potential[i - 1] + self.V_spike
                    # voltage is setback to resting potential
                    self.m_potential[i] = 0
                    numSpikes += 1
            # number of spikes
            spikes.append(numSpikes / self.time)
        plt.plot(inputCurrent, spikes)
        plt.xlabel("Input Current")
        plt.ylabel("Number of Spikes")
        plt.title("Frequency of spikes over different input currents")
        plt.show()


a = LIFModel(1)
a.plotNeuronSimulation()
a.plotNeuronSpikes()