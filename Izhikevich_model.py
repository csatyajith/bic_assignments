# implementation of the Izhikevich model
import numpy as np
import matplotlib.pyplot as plt


class Izhikevich_model:
    def __init__(self, I=1.5):
        self.I = I

        # all neuron properties
        self.totalTime = 500
        self.deltaT = 0.01
        self.time = np.arange(0, self.totalTime, self.deltaT)
        self.V = np.zeros(len(self.time))
        self.U = np.zeros(len(self.time))
        self.a_val = 0.02
        self.b_val = 0.2
        self.c_val = -65
        self.d_val = 2
        self.V_thresh = 30  # Threshold voltage
        self.V_spike = 0.5  # Spike delta

    def plotNeuronSimulation(self):
        self.V[0] = -55
        for i in range(1, len(self.time)):
            self.cal_membrane_potential(i,self.I)
            self.cal_membrane_recovery_potential(i)
            if self.V[i] >= self.V_thresh:
                self.update_membrane(i)
        return self.time,self.V,self.I


    def cal_membrane_potential(self,i,current):
        self.V[i] = self.V[i - 1] + self.deltaT * (
                0.04 * ((self.V[i - 1]) ** 2) + 5 * self.V[i - 1] + 140 - self.U[i - 1] + current)


    def cal_membrane_recovery_potential(self,i):
        self.U[i] = self.U[i - 1] + self.deltaT * (self.a_val * (self.b_val * self.V[i] - self.U[i - 1]))

    def update_membrane(self,i):
        self.V[i] = self.c_val
        self.U[i] = self.U[i] + self.d_val


    def plotNeuronSpikes(self):
        self.V[0] = -55
        n_spikes = []
        input_Current = np.arange(0, 7, 0.1)
        for current in input_Current:
            numSpikes = 0
            for i in range(1, len(self.time)):
                self.cal_membrane_potential(i,current)
                self.cal_membrane_recovery_potential(i)
                if self.V[i] >= self.V_thresh:
                    self.update_membrane(i)
                    numSpikes += 1
            n_spikes.append(numSpikes / self.totalTime)
        return input_Current,n_spikes


abc = Izhikevich_model(7)

t1,v1,i1=abc.plotNeuronSimulation()
plt.plot(t1,v1)
plt.xlabel("Time")
plt.ylabel("Membrane Potential in volts")
plt.title("Membrane Potential of a Neuron over Time with Input Current =" + str(i1))
plt.show()

input_Current,n_spikes=abc.plotNeuronSpikes()
plt.plot(input_Current, n_spikes)
plt.xlabel("Input Current")
plt.ylabel("Number of Spikes")
plt.title("Frequency of n_spikes over different input currents")
plt.show()
