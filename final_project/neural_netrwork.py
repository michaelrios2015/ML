import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def sig(x):
    return 1 / (1 + np.exp(-x))


# prunning function
def prunning(original, prune, error_supress, iter, k_iter):

    # first we need to make sure nothing that was pruned comes back
    original = original * prune

    # after k rounds check to see if a weight should be repressed
    if iter > 0 and iter % k_iter == 0:
        # so by here are weights are changed and we can see if any are small enough to turn off
        original = np.where(abs(original) > error_supress, original, 0)
        # if anything has been turned off it need to go over to our repressed matrix
        prune[original == 0] = 0

    return original, prune


###################################################################
#     easy input for testing
##################################################################

# # OR

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


###################################################################
#     actual input
##################################################################
data = np.genfromtxt("final_project/SPECTF.train", delimiter=",")
# data = np.genfromtxt("final_project/20.csv", delimiter=",")
# data = np.genfromtxt("final_project/30.csv", delimiter=",")
# data = np.genfromtxt("final_project/40.csv", delimiter=",")
# data = np.genfromtxt("final_project/50.csv", delimiter=",")

# print(data[0])
# I believe a random shuffle of the data is supposed to return better results
np.random.shuffle(data)
# classes are just the first column
classes = data[:, 0]
# features are everything else
features = np.delete(data, 0, axis=1)

###################################################################
#     this controls all of the parameters, how many layers, how many neurons in eash layer
#      epsilon, and epochs.. if there are more I connot think of them

##################################################################

# the length is how many layers we want, hidden and output layer, and how many neurons per layers
layers = np.array(
    [
        len(features[0]),
        len(features[0]),
        len(features[0]),
        1,
    ]
)


# our epsilon, epochs and error supression threshold
epsil = 0.001
epochs = 5000
# at what value to supress a weight
error_supress = 0.25
# after how many iterations to check if a weight should be supressed
k_iter = 50
###################################################################
#     nothing needs to be changed after this

##################################################################

################# WEIGHTS ###############################

weights = []
# so we will just keep an array of all ones, but elements can change to zero if we need to supress that rate
supress_weights = []

# loops through number a layers randomly pick weights and create my supreess weights matrix of all 1
for l in range(len(layers)):
    # first layer of weights is number of neurons times number of features
    if l == 0:
        temp = np.random.uniform(-1, 1, size=(layers[l], len(features[0])))
        temp_2 = np.ones((layers[l], len(features[0])))
    # all other layers of weights is number of neurons on that layer times number of neurons on pervious layers
    else:
        temp = np.random.uniform(-1, 1, size=(layers[l], layers[l - 1]))
        temp_2 = np.ones((layers[l], layers[l - 1]))
    # print(temp)
    weights.append(temp)
    supress_weights.append(temp_2)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
weights = np.array(weights, dtype=object)
supress_weights = np.array(supress_weights, dtype=object)


################# BIAS WEIGHTS ###############################

# pretty sure could have just included thse in the weights, I was a bit worried it would mess with the backward propagation
# and now I am running out of time.  So probably not the best code but works
bias_weights = []
supress_bias_weights = []

# loops through number a layers makes my bias weight and sets my supress matrix to all 1s
for l in range(len(layers)):
    # just one for each neuorn in the layer
    temp = np.random.uniform(-1, 1, layers[l])
    temp_2 = np.ones(layers[l])
    bias_weights.append(temp)
    supress_bias_weights.append(temp_2)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
bias_weights = np.array(bias_weights, dtype=object)
supress_bias_weights = np.array(supress_bias_weights, dtype=object)

################# NEURONS ###############################

neurons = []

# loops through number a layers
for l in range(len(layers)):
    # just one for each neuorn in the layer
    temp = np.zeros(layers[l])
    neurons.append(temp)

# convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
neurons = np.array(neurons, dtype=object)

# using this to store my mean squared errors
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

        ###################################################################
        #     forward propagation
        ##################################################################

        # loop through all layers
        for j in range(len(layers)):

            if j == 0:
                wtf = weights[j] * features[i]
            # after that weights times neurons
            else:
                wtf = weights[j] * neurons[j - 1]

            # sum of wights times features or neuorns depending on which level
            swtf = np.sum(wtf, axis=1)

            # Adding the biases
            total_sum = bias_weights[j] + swtf

            # not sure why my array is becoming more than 1 d
            if total_sum.ndim > 1:
                total_sum = total_sum[0]

            # sending total sums to sigmoid function, pretty sure this can be done more effeciently
            for k in range(len(total_sum)):
                neurons[j][k] = sig(total_sum[k])

        # calculate mean squared error
        error = error + 0.5 * pow((neurons[-1] - classes[i]), 2)
        errors.append(error)

        # there should be a way to put all of the back propagtion ine one for loop but to make my life easier I seperated out
        # the first step
        ###################################################################
        #     backward propagation top layer
        ##################################################################

        top = len(layers) - 1

        # step one calculate delta
        lil_delta = (1 - neurons[top]) * (classes[i] - neurons[top]) * neurons[top]

        # just the rest of the formula with the handy dandy use of arrays
        weights[top] = weights[top] + (epsil * lil_delta * neurons[top - 1])

        # probably a better way to combine these but this works for getting bias weights
        bias_weights[top] = bias_weights[top] + (epsil * lil_delta)

        # check for prunning any weights below the error supression level
        bias_weights[top], supress_bias_weights[top] = prunning(
            bias_weights[top], supress_bias_weights[top], error_supress, loop, k_iter
        )

        weights[top], supress_weights[top] = prunning(
            weights[top], supress_weights[top], error_supress, loop, k_iter
        )

        ###################################################################
        #     backward propagation all the other layers
        ##################################################################

        # now we loop backwards
        for bp in range(len(layers) - 2, -1, -1):

            # recalucalte error correction
            error_corr = lil_delta * weights[bp + 1]

            # we need to transform the array because we are now going backwards, at least i think that is why
            error_corr = error_corr.T

            # finally sum each row to get our new error correction
            error_corr = np.sum(error_corr, axis=1)

            # recaluclate lower (lil) delta
            lil_delta = (1 - neurons[bp]) * error_corr * neurons[bp]

            # recalculate bias weights, a little easier as onle one to each neuron
            bias_weights[bp] = bias_weights[bp] + epsil * lil_delta
            # print("bias_weights[bp]")
            # print(bias_weights[bp])

            # so we need to do some matrix multiplication but first we nee dto reshape the matrix, since there are multiple weights going to each neuron
            lil_delta = lil_delta.reshape(-1, 1)

            if bp >= 1:
                # here we are still in our hidden layers
                weights[bp] = weights[bp] + epsil * np.multiply(
                    lil_delta, neurons[bp - 1]
                )

            else:
                # here we are going back to the features
                weights[0] = weights[0] + epsil * np.multiply(lil_delta, features[i])

        # prinning the weight if they have gone below the prun threshold
        bias_weights[bp], supress_bias_weights[bp] = prunning(
            bias_weights[bp], supress_bias_weights[bp], error_supress, loop, k_iter
        )

        weights[bp], supress_weights[bp] = prunning(
            weights[bp], supress_weights[bp], error_supress, loop, k_iter
        )


print(weights)
print(bias_weights)

# just stuff for me to check how it did
print("last error")
print(errors[-1])

plt.plot(errors)
plt.show()

# using for confusion matrix
answers = []

###################################################################
#     seeing what we actually got right and wrong in training data
##################################################################
for i in range(len(features)):
    ###################################################################
    #     forward propagation
    ##################################################################

    # loop through all layers
    for j in range(len(layers)):

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

        total_sum = bias_weights[j] + swtf

        # not sure why my array is becoming more than 1 d
        if total_sum.ndim > 1:
            total_sum = total_sum[0]

        # sending total sums to sigmoid function
        for k in range(len(total_sum)):
            neurons[j][k] = sig(total_sum[k])

    if neurons[-1] > 0.5:
        answers.append(1)
    elif neurons[-1] < 0.5:
        answers.append(0)
    else:
        answers.append(neurons[j])


print("******************************")


print(answers)
print(classes)

cm = confusion_matrix(classes, answers)

print("******************************")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print("******************************")

###################################################################
#     checking test data
##################################################################

data = np.genfromtxt("final_project/SPECTF.test", delimiter=",")


classes = data[:, 0]


features = np.delete(data, 0, axis=1)

print(features[0])

answers = []
for i in range(len(features)):

    ###################################################################
    #     forward propagation
    ##################################################################

    # loop through all layers
    for j in range(len(layers)):

        if j == 0:
            wtf = weights[j] * features[i]
        # after that weights times neurons
        else:
            # print(neurons[j - 1])
            wtf = weights[j] * neurons[j - 1]

        # sum of wights times features or neuorns depending on which level
        swtf = np.sum(wtf, axis=1)

        total_sum = bias_weights[j] + swtf

        # not sure why my array is becoming more than 1 d
        if total_sum.ndim > 1:
            total_sum = total_sum[0]

        # sending total sums to sigmoid function
        for k in range(len(total_sum)):
            neurons[j][k] = sig(total_sum[k])

    if neurons[-1] > 0.5:
        answers.append(1)
    elif neurons[-1] < 0.5:
        answers.append(0)
    else:
        answers.append(neurons[j])


print("******************************")


print(answers)
print(classes)

cm = confusion_matrix(classes, answers)

print("******************************")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print("******************************")
