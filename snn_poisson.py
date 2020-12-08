import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from sklearn.datasets import load_digits


import matplotlib.pyplot as plt



#flattening to 1D arrays
#partitioning to training set, testing set, validation set
from sklearn.model_selection import train_test_split





def poissonEncoding(rates):
    """
    Transforms a 2D array of inputs representing an image into spike train representation utilizing Poisson method of
    encoding

    Params:
        rates: 2D array of inputs (in this case, pixel intensitiies corresponding to our image)
    Returns:
        n_rates: Normalized set of input pixel values (represents probabilities of a spike occurring at each dT)
        bin_spikes: 2D array of binary values representing timesteps in which spikes occur
        p_spikes: 2D array representing pixel intensities of spike train representation for plotting/computation purposes

    """
    dT = 1

    max = np.amax(rates)
    n_rates = (rates * dT) / max

    rand = np.random.uniform(0, 1, len(rates))
    # print(rand)
    # print(n_rates)
    bin_spikes = np.zeros(len(rates))
    pix_spikes = np.zeros(len(rates))

    for i in range(len(rates)):
        if n_rates[i] > rand[i]:
            bin_spikes[i] = 1
            pix_spikes[i] = max

    return n_rates, bin_spikes, pix_spikes


def train(X, Y, l_rt, init, epochs):
    weights = init

    for i in range(epochs):
        print("epoch: ", i)
        for x in range(len(X)):

            true_index = x % 10
            print("index: ", x, " true index: ", true_index)
            in_neurons = X[x]
            out_neurons = Y[x]

            in_n_rts, in_bins, in_pixs = poissonEncoding(in_neurons)

            print("input neurons: ", in_neurons)

            weight_adjs = np.zeros((len(in_neurons), 10))

            y_out = np.zeros((10, len(in_neurons)))
            y_out[true_index] = out_neurons.flatten()

            for pix in range(len(in_neurons)):
                for digit in range(10):
                    weight_adjs[pix][digit] = l_rt * in_bins[pix] * y_out[digit][pix]

            weights = weights + weight_adjs
            print("new weights: ", weights)

    return weights.T


def predict(test, weights, targ):
    n, spikes, p = poissonEncoding(test)
    outs = np.zeros(len(weights))
    for i in range(len(weights)):
        outs[i] = np.dot(spikes, weights[i])

    prediction = np.argmax(outs)

    print("predicted value: ", prediction)

    if (prediction == targ):
        return True
    else:
        return False

def sim(test, weights):
    correct = 0
    for i in range(len(test)):
        x = predict(test[i], weights, i % 10)
        print("prediction correct: ", x)
        print("--------------------------")
        if x:
            correct += 1
    return correct/len(test)


if __name__ == '__main__':
    digits = load_digits()
    plt.matshow(digits.images[1])
    plt.show()

    X_train, X_other, y_train, y_other = train_test_split(digits.data, digits.images, test_size=0.2, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_other, y_other, test_size=0.5, shuffle=False)
    print("Numbers to train: " + str(len(X_train)))
    print("Number of train's target: " + str(len(y_train)))
    print("Numbers to test: " + str(len(X_test)))
    print("Number of test's target: " + str(len(y_test)))
    print("Numbers to validate: " + str(len(X_val)))
    print("Number of validate's target: " + str(len(y_val)))

    print(X_val[0])

    print(digits.images[0])

    print(len(digits.images))


    init_weights = np.zeros((len(X_train[0]), 10))

    weights = train(X_train[0:20], y_train[0:20], 0.001, init_weights, 1)

    accuracy = sim(X_train[0:20], weights)
    print("accuracy: ", accuracy)








