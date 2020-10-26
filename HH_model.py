import numpy as np
import Gates as g
import matplotlib.pyplot as plt


class Hodgkin_Huxley:
    def __init__(self, current):
        self.I = current
        self.totaltime = 50
        self.deltaT = 0.001
        self.times = np.arange(0, self.totaltime, self.deltaT)
        self.V, self.INa, self.IK, self.Il, self.NStates, self.MStates, self.HStates = (np.zeros(len(self.times)) for i
                                                                                        in range(7))
        self.ENa, self.EK, self.El = 115, -12, 10.6
        self.gNa, self.gK, self.gl = 120, 36, 0.3
        self.m, self.n, self.h = (g.Gates() for i in range(3))
        self.C = 1
        self.eps = np.finfo(float).eps

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


hh = Hodgkin_Huxley(10)
hh.V[0] = -55
for i in range(1, len(hh.times)):
    hh.UpdateGateTimeConstants(hh.V[i - 1])
    hh.UpdateActionPotential(i)
    hh.updateGateVariables(i)

plt.plot(hh.times, hh.V)
plt.xlabel("Time in msec")
plt.ylabel("Membrane Voltage in milli volts")
plt.title("Membrane Voltage of a Neuron over Time with Input Current =" + str(hh.I))
plt.show()
