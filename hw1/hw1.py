import pandas as pd
import numpy as np
import math as m

import matplotlib.pyplot as plt


########################################################################
# the graph stuff hopefully done riht
#
################################################################
# # 1. Inspect the distribution of the speed feature for each class and determine if it follows a
# # SO I believe I need to make three arrays, one for each bird that only contains their speed
# #
# # Then see if it looks like
# # Gaussian or  Exponential or a Uniform
# #  Record this result as a comment in hw1.py .
# # so I have forgotten how the hsiotogram works

# dataIn = pd.read_csv("hw1\hw1Data.csv")
# dataNP = dataIn.to_numpy()

# finch = []
# duck = []
# sparrow = []
# raven = []

# # I think there is a much easier way to do this, but I am new to NP arrays
# for data in dataNP:
#     if data[0] == 0:
#         finch.insert(len(finch), data[1])
#     elif data[0] == 1:
#         duck.insert(len(duck), data[1])
#     elif data[0] == 2:
#         sparrow.insert(len(sparrow), data[1])
#     else:
#         raven.insert(len(raven), data[1])

# finch = np.array(finch)
# duck = np.array(duck)
# sparrow = np.array(sparrow)
# raven = np.array(raven)


# # print(finch)
# # print(duck)
# # print(sparrow)
# # print(raven)

# #################################################################
#
# #ANSWER FOR QUESTION 1
#
# ##################################################################


# plt.hist(finch)
# plt.show()
# # EXPONENTIAL!!!!!

# plt.hist(duck)
# plt.show()
# # going to say CONTINOUS

# plt.hist(sparrow)
# plt.show()
# # GAUSSIAN!!!

# plt.hist(raven)
# plt.show()
# # EXPONENTIAL!!!!!


########################################################################
# QUESTION 2
#
################################################################


# 2. Write a function called learnParams that takes in a data set and returns the learned
# lambda parameter for each class for the two features. Specifically, the function will be called as:
# params =learnParams(Data)
# where Data is a numpy array with shape (N,3) where N is the number of data points and
# params is a numpy array with shape (M,2) where there are M classes, params[i,0] is the
# lambda for the speed for class i and params[i,1] is the lambda for the chirp delay of class i.
# learnParams(np.array([[0,0.5,200],[1,0.7,130],[0,0.2,100],
# [1,2,10], [0,1.5,50],[1,4,20]])
# would return np.array([[1.36,0.009],[0.448,0.019]]) for params

# So at the moment very confused not sure how they are calculating lambda


def learnParams(data):

    # so we need a formula for lambda
    # used ln and then the derivative to get the formula for lambda
    # lambda = number of times a class is seen / sum of all the observations for that feature

    # We find a different lambda for each class

    # and then return those in an array based on class number and index

    # find out how many classes we have but looking at the class list and assuming the highest number is the hightest class
    num_classes = data[:, 0]
    # print(print(num_classes))
    top = int(max(num_classes)) + 1
    # print("top =", top)

    # these are two arrays both the length or our number of classes
    # this holds a count of how many times a class is seen
    classes = [0] * int(top)
    classes = np.array(classes)

    # this will hold the returned lambdas
    features = [[0.0, 0.0]] * int(top)
    features = np.array(features)
    # print(features[1, 1])

    # the clculations are pretty simple
    for a in data:
        classes[int(a[0])] = classes[int(a[0])] + 1
        # print(a[1])
        features[int(a[0]), 0] = features[int(a[0]), 0] + a[1]
        features[int(a[0]), 1] = features[int(a[0]), 1] + a[2]

    # vector math takes care of the return values :)
    return classes / features


print(
    learnParams(
        np.array(
            [
                [0, 0.5, 200],
                [1, 0.7, 130],
                [0, 0.2, 100],
                [1, 2, 10],
                [0, 1.5, 50],
                [1, 4, 20],
            ]
        )
    )
)

########################################################################
# QUESTION 3
#
################################################################
# 3. Write a function called learnPriors that takes in a data set and returns the prior
# probability of each class. Specifically, the function will be called as:
# priors=learnPriors(Data)
# where Data is a numpy array with shape (N,3) where N is the number of data points and
# priors is a numpy array with shape (M) where there are M classes, priors[i] is the
# estimated prior probability for class i .
# learnPriors(np.array([[0,0.5,200],[1,0.7,130],[0,0.2,100],
# [1,2,10], [0,1.5,50],[1,4,20]])
# would return np.array([0.5,0.5])

# seems nice and simple just count total of each species divide by total number of birds


def learnPriors(data):
    # so we need to go through and count how many of each bird(class) we have, is it safe to assume at least one of each class
    # I am going to say yes, do we first need to find how many classes we have?? probably not but that is how I will do it
    # for data in dataNP:

    # so this is probably unnessary but it helps me get a handle on the problem
    temp_list = []

    # I am just making a list of all the birds we see in the data
    for a in data:
        temp_list.insert(len(temp_list), a[0])

    temp_list = np.array(temp_list)
    # print(temp_list)
    # I am getting the max number in the list and assuming this is our last class of bird
    # and that we have seen all birds at least once
    top = temp_list.max()
    # print(top)
    # this is the total number of birds seen
    size = temp_list.size

    # this is a list in which we will keep a count of eaxh class of bird we see, index 0 for finch, etc, etc
    final = [0] * (int(top) + 1)
    # print(size)
    # print(final)

    # here we are actually counting
    for i in temp_list:
        # print(i)
        final[int(i)] = final[int(i)] + 1

    # finally we divide each class by the total size and those are our priors, at least I hope so
    return np.array(final) / size


print(
    learnPriors(
        np.array(
            [
                [0, 0.5, 200],
                [1, 0.7, 130],
                [0, 0.2, 100],
                [1, 2, 10],
                [0, 1.5, 50],
                [1, 4, 20],
            ]
        )
    )
)

########################################################################
# QUESTION 4
#
################################################################

# 4. Write a function called labelBayes that takes in posting times for multiple birds as well as
# the learned parameters for the likelihoods and prior, and return the most probable class for
# each bird. Specifically, the function will be called as:
# labelsOut = labelBayes(birdFeats,params,priors)
# where birdFeats is a numpy array of shape (K,2) containing the 2 features for K birds,
# params is a numpy array with shape (M,2) matching the description of the output for
# learnParams and priors is a numpy array with shape (M) matching the description of the
# output for learnPriors ; labelsOut is a numpy array with shape (K) containing the most
# probable label for each bird, where labelsOut[j] corresponds to birdFeats[j] .
# Labels are computed using the Exponential Bayes classifier!
# labelBayes(np.array([[0.5,5],[0.5,2],[2,8]]),
#  np.array([[0.7,0.2],[0.4,0.1]]),
#  np.array([0.4,0.6]))
# would return np.array([0,0,1])

# the formual for expontial distubtion


def exp_dist(lamb, x):
    return lamb / (m.e ** (lamb * x))


def labelBayes(birdFeats, params, priors):

    # for storing the winning class
    final = []

    # so if I undertood how to use vector math, and specifically vector math in pthon I might not need to use these for loops
    # right now I do, but hopefully I will learn the more efecient way :)
    for feat in birdFeats:
        # so we set the intial winner to zero and zero, so our very first run should beat that, the winner variable will not change
        # but that is fine
        pmax = 0
        winner = 0

        # then we have however many classes of birds, which is the length of params and priors
        for idx, param in enumerate(params):
            # then we apply the the params to each class and see which class is the biggest, we do not know how many classes but we only have two features

            # probability of class given first param and first feature measurment
            p1 = exp_dist(param[0], feat[0])

            # probability of class given 2nd param and 2nd feature measurment
            p2 = exp_dist(param[1], feat[1])

            # probabilties multipled with prior to get naive bayes
            ptotal = p1 * p2 * priors[idx]
            # if this total probablitly is bigger than the previous one make it the new pmax and remeber which class(winner) it is
            if ptotal > pmax:
                pmax = ptotal
                winner = idx

        # insert the winning class into our final array
        final.insert(len(final), winner)

    return np.array(final)


print(
    labelBayes(
        np.array([[0.5, 5], [0.5, 2], [2, 8]]),
        np.array([[0.7, 0.2], [0.4, 0.1]]),
        np.array([0.4, 0.6]),
    )
)


# birdFeats so this is teh speed and chipr thing, have as many as we have birds

# params the parameters lambda for the speed for class i and params[i,1]
# is the lambda for the chirp delay of class i. so for each bird fgeature I need to go through this and see which give me the maxium
# but also tie in the priors, these are as many as we have classes of birds so probably different from bordfeatures
