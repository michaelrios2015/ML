# 1. Write a function called neighborClassify that takes in a 1D numpy array of numbers
# (the heights of unknown animals) and a 2D numpy array containing the heights of known
# animals. The function will return a list of 0s and 1s – a 0 for each non-giraffe input and a 1
# for each giraffe input – using nearest neighbors classification (see below). Specifically, the
# function call must look like this:
# neighborClassify(featureArray, trainArray)
# featureArray will be a numpy array of shape (n) (where there are arbitrary number n
# animals to classify) and trainArray is a numpy array of shape (n,2 ) where each row
# contains first the height of a training animal and then its corresponding class (0 for non-giraffe,
# 1 for giraffe). Specifically, if featureArray=np.array([6,3,9]) and
# trainArray=np.array([[0.5,0], [1.5,0], [2.5,0], [4.5,1], [5,1],
# [7.5, 0], [8,1], [9.2,1]]), the function will return the list [1, 0, 1] .
# Classification is done by the nearest neighbors approach. Each test input is given the label of
# the nearest input in the training set. Below is a graphical example of this approach.

import numpy as np

featureArray = np.array([6, 3, 9])

trainArray = np.array(
    [[0.5, 0], [1.5, 0], [2.5, 0], [4.5, 1], [5, 1], [7.5, 0], [8, 1], [9.2, 1]]
)


def neighborClassify(featureArray, trainArray):

    # where we store our answers
    final = []

    for feature in featureArray:

        # we just assume that the feature will be closet to the first entry of the trainArray, we will then
        # check
        minDistance = abs(feature - trainArray[0][0])
        classChoosen = trainArray[0][1]

        # using brute force here, ee should use a binary search but I was not sure how much time I had
        for train in trainArray:
            # see if we get a new minDistance, note if we are tied on minDistance
            # then whichever class appeared first is choosen
            if minDistance > abs(feature - train[0]):
                # if so switch it
                minDistance = abs(feature - train[0])
                classChoosen = train[1]
        # insert the choosen class into our list of results
        final.insert(len(final), int(classChoosen))

    return final


# testing
print("testing Question 1")
print(neighborClassify(featureArray, trainArray))

################################################################################################################
################################################################################################################

# 2. Write a function called recalls that takes in a list of approximated class labels output by
# the classifier (threshClassify) and a list of true labels provided in the training set, and calculates
# the recall for each class – returning a n-element numpy array (presuming n classes) of the recall
# for each class (recallX =TrueClassX/AllClassX). Specifically, the function call must look like this:
# recalls(classifierOutput, trueLabels)
# If classifierOutput=[0,1,1,0,0,1,1,0,1] and
# trueLabels=[1,1,0,0,0,0,1,1,1], the function will return the numpy array
# np.array([0.5 , 0.6])
# For arbitrary input, you may presume the class labels are 0, 1, ..., n-1, e.g., when n=2, the labels
# are 0 and 1.

classifierOutput = [0, 1, 1, 0, 0, 1, 1, 0, 1]
trueLabels = [1, 1, 0, 0, 0, 0, 1, 1, 1]

# classifierOutput = [0, 0, 0, 1, 2, 2, 2, 7]
# trueLabels = [0, 0, 1, 1, 2, 1, 1, 7]


def recalls(classifierOutput, trueLabels):
    # so find how many classes there are
    # we are told class labels are 0, 1, ..., n-1,
    # and we assume trues guesses has at least one of last class, which seems like a safe assumption.
    top = int(max(trueLabels))
    # print("top ", top)

    # so we need to count all guesses and right guess of each class
    # the index will represent the class number

    # all times a class is guessed in classifierOutput
    classes = [0] * (top + 1)
    # all right guesses of a class in classifierOutput
    rightGuesses = [0] * (top + 1)

    # count the number of each class guess from classifierOutput
    # so we need to loop through classifierOutput
    for idx, classifier in enumerate(classifierOutput):

        # recallX = TrueClassX/AllClassX so that is right guesses/over total number of guesses for each class,
        # this ration goes into a new array in the order of the classes class 0 at 0 index etc.

        # did we guess right
        if classifier == trueLabels[idx]:
            # increminate right guesses at that index
            rightGuesses[classifier] = rightGuesses[classifier] + 1

        # counting the total times a class was guessed
        classes[classifier] = classes[classifier] + 1

    # print(classes)
    # print(rightGuesses)

    # if a class is never
    return np.array(rightGuesses) / np.array(classes)

    # count the number of right guess in each class from true lables,


print("testing Question 2")
print(recalls(classifierOutput, trueLabels))

# 3. Write a function called removeOnes that takes in a dataArray (n,2) numpy array and
# returns a (m,2) numpy array where any row with a 1 in the second row is removed.
# Specifically, the function call must look like this:
# expandedData=removeOnes(dataArray)
# If dataArray=np.array([[-4,2],[5,0],[3,0],[8,1],[10,0],[4,0],[2,1],[-2,2]])
# the function will return
# np.array([[-4,2],[5,0],[3,0], [10,0],[4,0] ,[-2,2]]) into
# expandedData .
# This function could be used for removing a class (class 1) not prioritized for learning.


dataArray = np.array(
    [[-4, 2], [5, 0], [3, 0], [8, 1], [10, 0], [4, 0], [2, 1], [-2, 2]]
)

# print(dataArray)


def removeOnes(dataArray):

    # we ar egoing to get the index for all entries with class 1
    ones = []

    for idx, data in enumerate(dataArray):
        if data[1] == 1:
            ones.insert(len(ones), idx)
    # print(ones)
    # Now we can just tell it which enties to delete
    return np.delete(dataArray, ones, 0)


print("testing question 3")
print(removeOnes(dataArray))
