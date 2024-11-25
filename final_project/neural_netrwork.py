import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv


def sig(x):
    return 1 / (1 + np.exp(-x))


###################################################################
#     easy input for testing
##################################################################
# XOR

# input_1 = [0, 0, 0]

# input_2 = [0, 1, 1]

# input_3 = [1, 0, 1]

# input_4 = [1, 1, 0]


# AND

# input_1 = [0, 0, 0]

# input_2 = [0, 1, 0]

# input_3 = [1, 0, 0]

# input_4 = [1, 1, 1]

# OR

# input_1 = [0, 0, 0]

# input_2 = [0, 1, 1]

# input_3 = [1, 0, 1]

# input_4 = [1, 1, 1]


# # seperating features and classes

# inputs = np.array([input_1, input_2, input_3, input_4])

# print(inputs)

# classes = inputs[:, 2]

# print(classes)

# features = np.delete(inputs, 2, axis=1)

# print(features)

errors = []


###################################################################
#     actual input
##################################################################
data = np.genfromtxt("final_project/SPECTF.train", delimiter=",")

print(data[0])

# np.random.shuffle(data)

classes = data[:, 0]

# print("classes")
# print(classes)

features = np.delete(data, 0, axis=1)

# print(features[0])

###################################################################
#     intialiozing weights and such
##################################################################

# probably a better way to do this
layers = 2

# how many neurons we want in layer 1
layer_1 = 3 * len(features[0])

# this is out output layer with only one or zero we just need one neuron
layer_2 = 1

# our epsilon
epsil = 0.1
# want a np array of this size

# weights layer 1, each layer represents the weights to one neuron
# so an array of arrays, the length is the number of neurons and then each indivual array is the number of features
wl1 = np.random.uniform(-1, 1, size=(layer_1, len(features[0])))
# a simple one for testing
# wl1 = np.array([[0.4, 0.8], [-0.4, 0.3]])

# print(wl1)

# weights layer 2, number of nuerons in layer two by the number of nueron in layer one
wl2 = np.random.uniform(-1, 1, size=(layer_2, layer_1))
# just for testing
# wl2 = np.array([[0.2, 0.3]])

# put all the weights into a new array to make it easier to reference
weights = np.array([wl1, wl2], dtype=object)

# print("weights")
# print(weights)

# bias weights layer 1, this is an earier array just 1-d the length equals number of neurons in layer 1
bwl1 = np.random.uniform(-1, 1, layer_1)
# simple test data
# bwl1 = np.array([0.2, 0.6])

# bias weights layer 2, see above just number of buerons in layer 2
bwl2 = np.random.uniform(-1, 1, layer_2)
# simple test data
# bwl2 = np.array([0.5])

# putting them in their own array to make it easier to refernce
bias_weights = np.array([bwl1, bwl2], dtype=object)

# neurons layer 1, start at zero
nl1 = np.zeros(layer_1)

# neurons layer 2, start at zero
nl2 = np.zeros(layer_2)

# our array of neurons
neurons = np.array([nl1, nl2], dtype=object)

# print("bias_weights[0]")
# print(bias_weights[0])

# however many epochs we want
for loop in range(10000):

    # helps me keep track of the mean sum error
    error = 0

    # going to loop through all the test data, or features
    for i in range(len(features)):

        # print("i")
        # print(i)
        ###################################################################
        #     forward propagation
        ##################################################################

        # loop through all layers
        for j in range(layers):

            # print("j")
            # print(j)
            # if it is the first loop weights times features
            if j == 0:
                wtf = weights[j] * features[i]
            # after that weights times neurons
            else:
                wtf = weights[j] * neurons[j - 1]
            # print("wtf")
            # print(wtf)

            # sum of wights times features or neuorns depending on which level
            swtf = np.sum(wtf, axis=1)
            # print("swtf")
            # print(swtf)

            # Adding the biases
            # print("bias_weights[j]")
            # print(bias_weights[j])
            total_sum = bias_weights[j] + swtf
            # print("total_sum")
            # print(total_sum)

            # sending total sums to sigmoid function
            for k in range(len(total_sum)):
                # that will be the value of our layer 1 neurons
                neurons[j][k] = sig(total_sum[k])

            # print("neurons[j]")
            # print(neurons[j])

        # calculate error??

        error = error + 0.5 * pow((neurons[-1] - classes[i]), 2)

        errors.append(error)

        # should be able to just put this in a for loop but honestly not sure how
        ###################################################################
        #     backward propagation layer 2
        ##################################################################

        # step one calculate delta

        delta_top = (1 - neurons[1]) * (classes[i] - neurons[1]) * neurons[1]

        # print("delta_top")
        # print(delta_top)

        # just the rest of the formula with the handy dandy use of arrays
        weights[1] = weights[1] + (epsil * delta_top * neurons[0])

        # probably a better way to combine these but this works for getting bias weights at least if that 1 value
        # is correct
        bias_weights[1] = bias_weights[1] + (epsil * delta_top)

        # print("layer 2 updated weights regular and bias")
        # print(weights[1])
        # print(bias_weights[1])
        # print("--------------------------")

        ###################################################################
        #     backward propagation layer 1
        ##################################################################

        # so here it gets a bit harder we have the delta

        # but this does seem to get us our new error correction thing
        error_corr = delta_top * weights[1]
        # print("error_corr")
        # print(error_corr)

        # I think this is the lower cas delta
        lil_deltal1 = (1 - neurons[0]) * error_corr * neurons[0]

        # print("lil_deltal1")
        # print(lil_deltal1)

        # so bias weights are easier as it is just one to each neuron
        # not entirely sure what i need to use lil_delta1[0] not entirely happy with that but seems to be working .. so
        # print("changed bias weights layer 1")
        bias_weights[0] = bias_weights[0] + epsil * lil_deltal1[0]
        # print("bias_weights[0]")
        # print(bias_weights[0])
        # print(bias_weights[0] + epsil * lil_deltal1)
        # so we need to do some matrix multiplication but first we nee dto reshape the matrix, since there are multiple weights going to each neuron
        lil_deltal1 = lil_deltal1.reshape(-1, 1)

        # print(lil_deltal1)

        ############## seems to be working
        # print("changed weights layer 1")
        weights[0] = weights[0] + epsil * np.multiply(lil_deltal1, features[i])
        # print(weights[0])

        # print("*******************************************")


print("last error")
print(errors[-1])

# plt.plot(errors)  # plotting by columns
# plt.show()
# print(errors)
# # eta

# print(weights)
# print(bias_weights)

# print("features")
# print(features)

# print("classes")
# print(classes)

answers = []

for i in range(len(features)):
    # print("i")
    # print(i)
    # print("classes[i]")
    # print(classes[i])
    ###################################################################
    #     forward propagation layer 1
    ##################################################################

    # wights times features level 1
    wtfl1 = weights[0] * features[i]

    # sum of wights times features level 1
    swtfl1 = np.sum(wtfl1, axis=1)
    # print("swtfl1")
    # print(swtfl1)

    total_sum = 0
    # sum of weight + biases
    # print("bias_weights[0]")
    # print(bias_weights[0])
    total_sum = bias_weights[0] + swtfl1
    # print("total_sum")
    # print(total_sum)

    # sending total sums to sigmoid function
    for j in range(len(total_sum)):
        # that will be the value of our layer 1 neurons
        neurons[0][j] = sig(total_sum[j])

    # print("neurons[0]")
    # print(neurons[0])

    ###################################################################
    #     forward propagation layer 2
    ##################################################################

    # layer 2 weights times neurons of layer 1
    lw2n1 = neurons[0] * weights[1]

    # print(lw2n1)

    # sum of wights times features level 1
    swtfl2 = np.sum(lw2n1, axis=1)
    # print(swtfl2)

    # sum of weight + biases
    total_sum_2 = bias_weights[1] + swtfl2
    # print(total_sum_2)

    for k in range(len(total_sum_2)):
        neurons[1][k] = sig(total_sum_2[k])

    if neurons[1] > 0.5:
        answers.append(1)
    elif neurons[1] < 0.5:
        answers.append(0)
    else:
        answers.append(neurons[1])
    # print("neurons[1]")
    # print(neurons[1])
    # print("classes[i]")
    # print(classes[i])

    print("******************************")


print(answers)
print(classes)
