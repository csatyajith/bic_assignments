import numpy as np
import Gates as g
import matplotlib.pyplot as plt


class Hodgkin_Huxley:
    def __init__(self, current):
        self.I = current
        self.totaltime = 50
        self.deltaT = 0.01
        self.times = np.arange(0, self.totaltime, self.deltaT)
        self.V, self.INa, self.IK, self.Il, self.NStates, self.MStates, self.HStates = (np.zeros(len(self.times)) for i
                                                                                        in range(7))
        self.ENa, self.EK, self.El = 115, -12, 10.6  # Equilibrium potentials of Na, K and Cl
        self.gNa, self.gK, self.gl = 120, 36, 0.3  # Conductances of Na, K, Cl
        self.m, self.n, self.h = (g.Gates() for i in range(3))  # Gating variables
        self.C = 1  # Membrane Capacitance
        self.V_thresh = 55

    def plotNeuronSimulation(self):
        self.V[0] = -55
        numSpikes = 0
        for i in range(1, len(self.times)):
            self.UpdateGateTimeConstants(self.V[i - 1])
            self.UpdateActionPotential(i)
            self.updateGateVariables(i)
            if self.V[i] >= self.V_thresh:
                numSpikes += 1
        fig, axs = plt.subplots(2)
        fig.suptitle("Membrane Voltage of a Neuron over Time with Input Current =" + str(self.I), y=1)
        axs[0].plot(self.times, self.V)
        axs[1].plot(self.times, self.MStates, 'tab:orange', label='m')
        axs[1].plot(self.times, self.NStates, 'tab:green', label='n')
        axs[1].plot(self.times, self.HStates, 'tab:red', label='h')
        plt.legend(loc=1)
        plt.show()
        return numSpikes

    def UpdateGateTimeConstants(self, V):
        self.m.alpha = .1 * ((25 - V) / (np.exp((25 - V) / 10) - 1))
        self.m.beta = 4 * np.exp(-V / 18)
        self.n.alpha = .01 * ((10 - V) / (np.exp((10 - V) / 10) - 1))
        self.n.beta = .125 * np.exp(-V / 80)
        self.h.alpha = .07 * np.exp(-V / 20)
        self.h.beta = 1 / (np.exp((30 - V) / 10) + 1)

    def UpdateActionPotential(self, i):
        self.INa = self.gNa * np.power(self.MStates[i - 1], 3) * self.HStates[i - 1] * (self.V[i - 1] - self.ENa)
        self.IK = self.gK * np.power(self.NStates[i - 1], 4) * (self.V[i - 1] - self.EK)
        self.Il = self.gl * (self.V[i - 1] - self.El)
        self.V[i] = self.V[i - 1] + self.deltaT * ((self.I - self.INa - self.IK - self.Il) / self.C)

    def updateGateVariables(self, i):
        self.MStates[i] = self.m.updateGateState(self.deltaT)
        self.NStates[i] = self.n.updateGateState(self.deltaT)
        self.HStates[i] = self.h.updateGateState(self.deltaT)


input_Current = []
total_spikes = []
for cur in range(12):
    hh = Hodgkin_Huxley(cur)
    n_spikes = hh.plotNeuronSimulation()
    input_Current.append(cur)
    total_spikes.append(n_spikes / (1000 * hh.totaltime))
plt.plot(input_Current, total_spikes)
plt.xlabel("Input Current")
plt.ylabel("Number of Spikes")
plt.title("Frequency of n_spikes over different input currents")
plt.show()
