import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

input_1 = [0, 0, 0]

input_2 = [0, 1, 1]

input_3 = [1, 0, 1]

input_4 = [1, 1, 1]


# seperating features and classes

inputs = np.array([input_1, input_2, input_3, input_4])

print(inputs)

classes = inputs[:, 2]

print(classes)

features = np.delete(inputs, 2, axis=1)

print(features)


# ###################################################################
# #     actual input
# ##################################################################
# data = np.genfromtxt("final_project/SPECTF.train", delimiter=",")

# # data = np.genfromtxt("final_project/20.csv", delimiter=",")

# # data = np.genfromtxt("final_project/30.csv", delimiter=",")

# # data = np.genfromtxt("final_project/40.csv", delimiter=",")

# # data = np.genfromtxt("final_project/50.csv", delimiter=",")

# print(data[0])

# np.random.shuffle(data)

# # print("data")
# # print(data[10])

# classes = data[:, 0]

# # print("classes")
# # print(classes)

# features = np.delete(data, 0, axis=1)

# print(features[0])

###################################################################
#     this controls all of the parameters, how many layers, how many neurons in eash layer
#      epsilon, and epochs.. if there are more I connot think of them

##################################################################

# the length is how many layers we want, hidden and output layer, and how many neurons per layers
layers = np.array(
    [
        len(features[0]),
        1,
    ]
)

# print(layers)
# print(len(layers))


# our epsilon
epsil = 0.002
# want a np array of this size

epochs = 1

###################################################################
#     nothing needs to be changed after this

##################################################################
################# WEIGHTS ###############################

weights = []

# loops through number a layers
for l in range(len(layers)):
    # first layer of weights is neaturs time number of neurons
    if l == 0:
        temp = np.random.uniform(-1, 1, size=(layers[l], len(features[0])))
    # all other layers of weights is number of neurons on that layer times number of neurons on pervious layers
    else:
        temp = np.random.uniform(-1, 1, size=(layers[l], layers[l - 1]))
    # print(temp)
    weights.append(temp)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
weights = np.array(weights, dtype=object)

# print(weights)

# simple test weights
# weights layer 1, each layer represents the weights to one neuron
# wl1 = np.array([[0.1, 0.7], [0.9, -0.1]])
# # wl2 = np.array([[0.4, 0.8], [-0.4, 0.3]])
# # wl3 = np.array([[0.2, 0.3]])
# # put all the weights into a new array to make it easier to reference
# weights = np.array([wl1, wl2, wl3], dtype=object)
# print("weights")
# print(weights)

################# BIAS WEIGHTS ###############################

bias_weights = []

# loops through number a layers
for l in range(len(layers)):
    # just one for each neuorn in the layer
    temp = np.random.uniform(-1, 1, layers[l])
    bias_weights.append(temp)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
bias_weights = np.array(bias_weights, dtype=object)

print("bias_weights")
print(bias_weights)

# so simple test data
# # bias weights layer 1, this is an earier array just 1-d the length equals number of neurons in layer 1
# # simple test data
# # bwl1 = np.array([0.3, 0.4])
# # bwl2 = np.array([-0.2, -0.6])
# # bwl3 = np.array([0.5])
# # putting them in their own array to make it easier to refernce
# bias_weights = np.array([bwl1, bwl2, bwl3], dtype=object)
# print("bias_weights")
# print(bias_weights)

################# NEURONS ###############################


neurons = []

# loops through number a layers
for l in range(len(layers)):
    # just one for each neuorn in the layer
    temp = np.zeros(layers[l])
    neurons.append(temp)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
neurons = np.array(neurons, dtype=object)

print("neurons")
print(neurons)

errors = []

###################################################################
#     Starting the Network
##################################################################

# however many epochs we want
for loop in range(epochs):

    # helps me keep track of the mean sum error
    error = 0

    # going to loop through all the test data, or features
    for i in range(len(features)):

        # print("features[i]")
        # print(features[i])

        # print("i")
        # print(i)
        ###################################################################
        #     forward propagation
        ##################################################################

        # loop through all layers
        for j in range(len(layers)):

            # print("------------")
            # print("j")
            # print(j)
            # if it is the first loop weights times features
            if j == 0:
                wtf = weights[j] * features[i]
            # after that weights times neurons
            else:
                # print(neurons[j - 1])
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

            # not sure why my array is becoming more than 1 d
            if total_sum.ndim > 1:
                total_sum = total_sum[0]
            # print("total_sum")
            # print(total_sum)

            # sending total sums to sigmoid function
            for k in range(len(total_sum)):
                # print("j")
                # print(j)
                # print("k")
                # print(k)
                # print(neurons[j][k])
                # that will be the value of our layer 1 neurons
                neurons[j][k] = sig(total_sum[k])

            # print("neurons[j]")
            # print(neurons[j])

            # print("------------")

        # calculate error??

        error = error + 0.5 * pow((neurons[-1] - classes[i]), 2)

        errors.append(error)
        # print("neurons")
        # print(neurons)

        # should be able to just put this in a for loop but honestly not sure how
        ###################################################################
        #     backward propagation top layer
        ##################################################################

        # step one calculate delta

        top = len(layers) - 1

        lil_delta = (1 - neurons[top]) * (classes[i] - neurons[top]) * neurons[top]
        # print("delta_top")
        # print(delta_top)

        # just the rest of the formula with the handy dandy use of arrays
        weights[top] = weights[top] + (epsil * lil_delta * neurons[top - 1])

        # probably a better way to combine these but this works for getting bias weights at least if that 1 value
        # is correct
        bias_weights[top] = bias_weights[top] + (epsil * lil_delta)

        # print("top updated weights regular and bias")
        # print(weights[top])
        # print(bias_weights[top])
        # print("--------------------------")

        ###################################################################
        #     backward propagation all the other layers
        ##################################################################

        # so here it gets a bit harder we have the delta

        # but this does seem to get us our new error correction thing

        for bp in range(len(layers) - 2, -1, -1):
            # print("bp")
            # print(bp)

            # so this is the second to
            # if bp == len(layers) - 2:
            error_corr = lil_delta * weights[bp + 1]
            # print("error_corr")
            # print(error_corr)
            # else:
            #     error_corr = lil_deltal * weights[bp + 1]

            error_corr = error_corr.T
            # print("error_corr")
            # print(error_corr)

            # finally sum each row to get our new lower case delta
            error_corr = np.sum(error_corr, axis=1)

            # print("error_corr")
            # print(error_corr)

            # I think this is the lower cas delta
            lil_delta = (1 - neurons[bp]) * error_corr * neurons[bp]
            # print("lil_deltal")
            # print(lil_delta)

            # so bias weights are easier as it is just one to each neuron
            # not entirely sure what i need to use lil_delta1[0] not entirely happy with that but seems to be working .. so
            # print("changed bias weights layer 1")
            bias_weights[bp] = bias_weights[bp] + epsil * lil_delta
            # print("bias_weights[bp]")
            # print(bias_weights[bp])

            # so we need to do some matrix multiplication but first we nee dto reshape the matrix, since there are multiple weights going to each neuron
            lil_delta = lil_delta.reshape(-1, 1)
            # print("lil_delta")
            # print(lil_delta)

            # print("changed weights layer")
            # print(bp - 1)
            if bp >= 1:
                ############## seems to be working
                weights[bp] = weights[bp] + epsil * np.multiply(
                    lil_delta, neurons[bp - 1]
                )

            else:
                ############## seems to be working
                # print("changed weights layer 1")
                weights[0] = weights[0] + epsil * np.multiply(lil_delta, features[i])
                # print(weights[0])
            # print("changed weights layer")
            # print(bp)
            # print(weights[bp])
            # print("--------------------------")


print("last error")
print(errors[-1])

# plt.plot(errors)  # plotting by columns
# plt.show()
# # # print(errors)

# # print(weights)
# # print(bias_weights)

# # print("features")
# # print(features)

# # print("classes")
# # print(classes)

# answers = []


# ###################################################################
# #     seeing what we actually got right and wrong in training data
# ##################################################################
# for i in range(len(features)):
#     ###################################################################
#     #     forward propagation
#     ##################################################################

#     # loop through all layers
#     for j in range(len(layers)):

#         if j == 0:
#             wtf = weights[j] * features[i]
#         # after that weights times neurons
#         else:
#             # print(neurons[j - 1])
#             wtf = weights[j] * neurons[j - 1]
#         # print("wtf")
#         # print(wtf)

#         # sum of wights times features or neuorns depending on which level
#         swtf = np.sum(wtf, axis=1)

#         total_sum = bias_weights[j] + swtf

#         # not sure why my array is becoming more than 1 d
#         if total_sum.ndim > 1:
#             total_sum = total_sum[0]

#         # sending total sums to sigmoid function
#         for k in range(len(total_sum)):
#             neurons[j][k] = sig(total_sum[k])

#     if neurons[-1] > 0.5:
#         answers.append(1)
#     elif neurons[-1] < 0.5:
#         answers.append(0)
#     else:
#         answers.append(neurons[j])


# print("******************************")


# print(answers)
# print(classes)

# cm = confusion_matrix(classes, answers)

# print("******************************")
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()
# print("******************************")

# ###################################################################
# #     checking test data
# ##################################################################

# data = np.genfromtxt("final_project/SPECTF.test", delimiter=",")


# classes = data[:, 0]


# features = np.delete(data, 0, axis=1)

# print(features[0])

# answers = []
# for i in range(len(features)):

#     ###################################################################
#     #     forward propagation
#     ##################################################################

#     # loop through all layers
#     for j in range(len(layers)):

#         if j == 0:
#             wtf = weights[j] * features[i]
#         # after that weights times neurons
#         else:
#             # print(neurons[j - 1])
#             wtf = weights[j] * neurons[j - 1]

#         # sum of wights times features or neuorns depending on which level
#         swtf = np.sum(wtf, axis=1)

#         total_sum = bias_weights[j] + swtf

#         # not sure why my array is becoming more than 1 d
#         if total_sum.ndim > 1:
#             total_sum = total_sum[0]

#         # sending total sums to sigmoid function
#         for k in range(len(total_sum)):
#             neurons[j][k] = sig(total_sum[k])

#     if neurons[-1] > 0.5:
#         answers.append(1)
#     elif neurons[-1] < 0.5:
#         answers.append(0)
#     else:
#         answers.append(neurons[j])


# print("******************************")


# print(answers)
# print(classes)

# cm = confusion_matrix(classes, answers)

# print("******************************")
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()
# print("******************************")
