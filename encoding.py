import numpy as np
import matplotlib.pyplot as plt
import math
from lif_neuron import LIFNeuron

inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
weights = np.zeros((1, 2))
time = np.arange(0, 100, 1)
potential_list = []
for i in range(4):
    potential_list.append([])

A_plus = 0.8
A_minus = 0.3
tau_plus = 8
tau_minus = 5
sliding_window = np.arange(-15 - 1, 15 + 1, 1)


def weight_init():
    for i in range(1):
        for j in range(2):
            weights[i][j] = np.random.uniform(0, 1)


def update_weights(w, t):
    if t > 0:
        deltaW = -A_plus * np.exp(-float(t) / tau_plus)
    if t <= 0:
        deltaW = A_minus * np.exp(float(t) / tau_minus)

    if deltaW < 0:
        return w + 0.5 * deltaW * w
    elif deltaW > 0:
        return w + 0.5 * deltaW * (1 - w)


def rate_encoding(x):
    spike_train = []
    for y in range(2):
        temp = np.zeros((100,))
        freq = int(inp[x][y] * 30 + 10)
        freq = math.ceil(600 / freq)
        i = 0
        while i < 100:
            temp[i] = 1
            i = i + freq
        spike_train.append(temp)

    return spike_train


def lif(train, n):
    # print(weights[0])
    print('train', train)
    for t in time:
        if op_neuron.t_inactive < t:
            op_neuron.membrane_potential += np.dot(weights[0], train[:, t])

        potential_list[n].append(op_neuron.membrane_potential)
        print('membrane pot', op_neuron.membrane_potential)
        if op_neuron.membrane_potential >= op_neuron.threshold_v:
            op_neuron.t_inactive = t + op_neuron.ref_period
            print('time inactive at t =', op_neuron.t_inactive, t)
            op_neuron.membrane_potential = op_neuron.resting_v
            for i in range(2):
                for t1 in sliding_window:
                    if 0 <= t + t1 < 100 and t1 != 0:
                        if train[i][t + t1] == 1:
                            # print('weight change by' + str(update_weights(weights[0][i], t1))+ 'for neuron '+str(i))
                            weights[0][i] = update_weights(weights[0][i], t1)


op_neuron = LIFNeuron()
weight_init()

for k in range(10):
    print('Iteration ', k)
    print('weights', weights)
    for i in range(4):
        train = np.array(rate_encoding(i))
        op_neuron.reset()
        lif(train, i)

print('Final weights', weights)

for i in range(4):
    totalTime = np.arange(0, len(potential_list[i]), 1)
    Pth = []
    for j in range(len(totalTime)):
        Pth.append(op_neuron.threshold_v)
    # plotting
    axes1 = plt.gca()
    plt.plot(totalTime, Pth, 'r')
    plt.plot(totalTime, potential_list[i])
    plt.show()

colorCodes = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

plt.eventplot(potential_list, color=colorCodes)
plt.title('Spike raster plot')
plt.xlabel('Spike')
plt.ylabel('Neuron')
plt.show()
