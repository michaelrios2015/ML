import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


###################################################################
#     THE FUNCTIONS!!
##################################################################


# a small sigmoid function I believe a lambada function would have probably been better
def sig(x):
    return 1 / (1 + np.exp(-x))


# prunning function
def prunning(original, prune, supress_threshold, iter, k_iter):
    # print("============================")
    # print(original)
    # first we need to make sure nothing that was pruned comes back
    original = original * prune

    # after k rounds check to see if a weight should be repressed
    if iter > 0 and iter % k_iter == 0:
        # so by here are weights are changed and we can see if any are small enough to turn off
        original = np.where(abs(original) < supress_threshold, 0, original)
        # print("============================ if ")
        # print(original)
        # if anything has been turned off it need to go over to our repressed matrix
        prune[original == 0] = 0
    # print("============================")
    # print(original)

    return original, prune


###################################################################
#     OUR MAN FUNTION THE NEURAL NETWORK
##################################################################


# if I had more time I would sepearte this out into a bunch more smaller functions, but it seems to work
# and is better than just the large chunck of code I had before
def neural_net(
    _epochs,
    _epsil,
    _features,
    _classes,
    _layers,
    _supress_threshold,
    _k_iter,
):

    # we start off by creating all the weights (bias and regural) and the neurons
    #################  WEIGHTS ###############################
    weights = []
    # so we will just keep an array of all ones, but elements can change to zero if we need to supress that rate
    supress_weights = []

    # loops through number a layers randomly pick weights and create my supreess weights matrix of all 1s
    for l in range(len(_layers)):
        # first layer of weights is number of neurons times number of features
        if l == 0:
            temp = np.random.uniform(-1, 1, size=(_layers[l], len(features[0])))
            temp_2 = np.ones((_layers[l], len(_features[0])))
        # all other layers of weights is number of neurons on that layer times number of neurons on pervious layers
        else:
            temp = np.random.uniform(-1, 1, size=(_layers[l], _layers[l - 1]))
            temp_2 = np.ones((_layers[l], _layers[l - 1]))
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
    for l in range(len(_layers)):
        # just one for each neuorn in the layer
        temp = np.random.uniform(-1, 1, _layers[l])
        temp_2 = np.ones(_layers[l])
        bias_weights.append(temp)
        supress_bias_weights.append(temp_2)

    # convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
    bias_weights = np.array(bias_weights, dtype=object)
    supress_bias_weights = np.array(supress_bias_weights, dtype=object)

    ################# NEURONS ###############################

    neurons = []

    # loops through number a layers
    for l in range(len(_layers)):
        # just one for each neuorn in the layer
        temp = np.zeros(_layers[l])
        neurons.append(temp)

    # convert it to an oddly shaped np array not sure if this is the best way to do this but seems to work
    neurons = np.array(neurons, dtype=object)

    # using this to store my mean squared errors

    # helps me se how wer are doing
    errors = []
    # however many epochs we want

    for loop in range(_epochs):

        # helps me keep track of the mean sum error
        error = 0

        # going to loop through all the test data, or features
        for i in range(len(_features)):

            ###################################################################
            #     forward propagation, ideally this would be it's own function
            ##################################################################

            # loop through all layers
            for j in range(len(_layers)):

                if j == 0:
                    wtf = weights[j] * _features[i]
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
            error = error + 0.5 * pow((neurons[-1] - _classes[i]), 2)
            errors.append(error)

            # there should be a way to put all of the back propagtion ine one for loop but to make my life easier I seperated out
            # the first step
            ###################################################################
            #     backward propagation top layer, also ideally it's own function
            ##################################################################

            top = len(_layers) - 1

            # step one calculate delta
            lil_delta = (1 - neurons[top]) * (_classes[i] - neurons[top]) * neurons[top]

            # just the rest of the formula with the handy dandy use of arrays
            weights[top] = weights[top] + (_epsil * lil_delta * neurons[top - 1])

            # probably a better way to combine these but this works for getting bias weights
            bias_weights[top] = bias_weights[top] + (_epsil * lil_delta)

            # check for prunning any weights below the error supression level
            bias_weights[top], supress_bias_weights[top] = prunning(
                bias_weights[top],
                supress_bias_weights[top],
                _supress_threshold,
                loop,
                _k_iter,
            )

            weights[top], supress_weights[top] = prunning(
                weights[top], supress_weights[top], _supress_threshold, loop, _k_iter
            )

            ###################################################################
            #     backward propagation all the other layers
            ##################################################################

            # now we loop backwards
            for bp in range(len(_layers) - 2, -1, -1):

                # recalucalte error correction
                error_corr = lil_delta * weights[bp + 1]

                # we need to transform the array because we are now going backwards, at least i think that is why
                error_corr = error_corr.T

                # finally sum each row to get our new error correction
                error_corr = np.sum(error_corr, axis=1)

                # recaluclate lower (lil) delta
                lil_delta = (1 - neurons[bp]) * error_corr * neurons[bp]

                # recalculate bias weights, a little easier as onle one to each neuron
                bias_weights[bp] = bias_weights[bp] + _epsil * lil_delta
                # print("bias_weights[bp]")
                # print(bias_weights[bp])

                # so we need to do some matrix multiplication but first we nee dto reshape the matrix, since there are multiple weights going to each neuron
                lil_delta = lil_delta.reshape(-1, 1)

                if bp >= 1:
                    # here we are still in our hidden layers
                    weights[bp] = weights[bp] + _epsil * np.multiply(
                        lil_delta, neurons[bp - 1]
                    )

                else:
                    # here we are going back to the features
                    weights[0] = weights[0] + _epsil * np.multiply(
                        lil_delta, _features[i]
                    )

            # prinning the weight if they have gone below the prun threshold
            bias_weights[bp], supress_bias_weights[bp] = prunning(
                bias_weights[bp],
                supress_bias_weights[bp],
                _supress_threshold,
                loop,
                _k_iter,
            )

            weights[bp], supress_weights[bp] = prunning(
                weights[bp], supress_weights[bp], _supress_threshold, loop, _k_iter
            )

    # fianlly what we will return
    return weights, bias_weights, neurons, errors


###################################################################
#     Input, ideally a function but it is so small
##################################################################
data = np.genfromtxt("final_project/SPECTF.train", delimiter=",")

# print(data[0])
# I believe a random shuffle of the data is supposed to return better results
np.random.shuffle(data)
# classes are just the first column
classes = data[:, 0]
# features are everything else
features = np.delete(data, 0, axis=1)

###################################################################
#     USER INPUT --- entire how many layers, how many neurons per layer, epsilon, epochs,
#     supression threshold (supress_threshold), and number of epochs at which to check for supression (k_iter)
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
# at what value to supress a weight, really
supress_threshold = 0.1
# after how many iterations to check if a weight should be supressed
k_iter = 5
###################################################################
#     nothing needs to be changed after this

##################################################################


# calling your function
weights, bias_weights, neurons, errors = neural_net(
    epochs,
    epsil,
    features,
    classes,
    layers,
    supress_threshold,
    k_iter,
)

# just used this to make sure weights were being supressed
# print(weights)
# print(bias_weights)

# just stuff for me to check how it did
print("last error")
print(errors[-1])

plt.plot(errors)
plt.show()

# using for confusion matrix
answers = []


###################################################################
#     seeing what we actually got right and wrong in training data
#     so this is just the forward propagtion portion of the code with out final paramters
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


# just showing how we did
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

# so we get the test data
data = np.genfromtxt("final_project/SPECTF.test", delimiter=",")
# diving it up bewteen classes and features
classes = data[:, 0]
features = np.delete(data, 0, axis=1)
# print(features[0])

answers = []
# again just using the forward propagtion portion of the code to see how we did, again this should be it's
# own function but for the moment it's fine
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

# our output to see hwo we did

print("******************************")
print(answers)
print(classes)

cm = confusion_matrix(classes, answers)

print("******************************")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print("******************************")
