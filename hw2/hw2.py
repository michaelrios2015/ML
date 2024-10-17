import scipy.io
import numpy as np

# looking at the data
mat = scipy.io.loadmat("hw2\hw2data.mat")

# print(mat["fullData"][0])

# print(len(mat["fullData"]))


# 1. Write a function called kernelClassify that will return the class +1 or -1 based on
# the input of a single data point to classify, a set of support vectors, their corresponding
# 𝜶 values and the b constant offset. Specifically, the function will be called as:
# kernelClassify(dataPt, suppVecs, alphas, b)
# where dataPt is a numpy array with shape (M), suppVecs is a numpy array with
# shape (N,M+1) (the first M columns are features, the last column is +1 or -1 label),
# alphas is a numpy array with shape (N) with non-negative 𝜶 values matching each
# vector in suppVecs , and b is a single number offset. The function will return a single
# number +1 or -1 to indicate whether the dataPt is in class +1 or -1.
# This function will use the kernel: 𝐾(𝒙, 𝒗) = (𝒙𝑇𝒗 + 1)4
# .
# Classification of a given vector 𝒗 will be performed using 𝑏 + ∑ 𝛼𝑖𝑦𝑖𝐾 (𝒙𝒊 𝑇𝑖 , 𝒗) , testing
# whether the value is above +1 or below -1.
# If the sum is between +1 and -1 (it is TOO CLOSE to the separator), we recommend you
# output 0 as your answer (but we will not take off points if you output 1 or -1 in this
# homework.


# can use fake data for this hopefully relatively easy

dataPt = np.array([0, 0, 0, 0, 0, 1])
suppVecs = np.array(
    [
        [-3.5, -1, -3, 1, 0.5, 2, 1],
        [-1.5, -0.5, -4, -1, 0, 3, 1],
        [3, 0, 4, 0, 0.5, 0.0, -1],
    ]
)
alphas = np.array([0.5, 0.7, 1.2])
b = 12.5


def kernel(x, v):

    # print(x)
    return (np.dot(x, v) + 1) ** 4


# print(kernel(np.array([-3.5, -1, -3, 1, 0.5, 2]), dataPt))


# it's something
def kernelClassify(dataPt, suppVecs, alphas, b):

    # so let's do the summation part first

    total = 0

    for i in range(0, len(suppVecs)):

        # print(suppVecs[i][-1])
        # print(suppVecs[i][:-1])
        total += alphas[i] * suppVecs[i][-1] * kernel(suppVecs[i][:-1], dataPt)
        # print(total)

    total += b

    # sould it be >= 1??
    if total > 1:
        return 1
    elif total < -1:
        return -1
    else:
        return 0


# print(kernelClassify(dataPt, suppVecs, alphas, b))

###############################################################################################################################

# 2. Write a function computeW to compute the linear separator vector w given a data set
# x, corresponding class labels y, and corresponding weights 𝜶. Specifically, the function
# will be called as:
# computeW(alphas, labels, dataSet)
# where alphas is a numpy array with shape (N) ), labels is a numpy array with shape
# (N), and dataSet is a numpy array with shape (N,M. This means there are N data
# points and M features. The function will return the shape (M) numpy array computed by
# 𝒘 = ∑ 𝛼𝑖𝑦𝑖𝒙𝒊

# some test data, need to make numpy array
alphas = np.array([0.5, 0.7, 1.2])
labels = np.array([1, 1, -1])
dataSet = np.array(
    [[-3.5, -1, -3, 1, 0.5, 2], [-1.5, -0.5, -4, -1, 0, 3], [3, 0, 4, 0, 0.5, 0]]
)


# essentially just question 8, multiple vectors by
def computeW(alphas, labels, dataSet):

    # this will be our return array
    n = len(dataSet[0])
    total = np.zeros(n)

    # loop through all of them
    for i in range(0, len(dataSet)):
        # mutiple vector by alpha and label
        dataSet[i] = alphas[i] * labels[i] * dataSet[i]
        # add to total
        total = total + dataSet[i]

    return total


# print(computeW(alphas, labels, dataSet))

# will start with fake data this should be the easisest

################################################################################################################################

# 3. Write a function called learnLam that will return the 𝜆’s derived during linear SVM
# learning based on the input data points and a specified number of iterations.
# Specifically, the function will be called as:
# learnLam(dataTrain, iters)
# where dataTrain is a numpy array with shape/size (N,M+1) - the first M columns are
# features, the last column is +1 or -1 label; iter is a single number indicating the
# number of times to loop through all data points before returning w.
# The function will return lambdaOut, b, and loss . lambdaOut is a numpy array
# with shape/size (N) where row k contains the non-negative lambda corresponding to
# the k-1th data point in dataTrain. b is a single real number offset. loss is a numpy
# array with shape/size (len(iter)) where the entry at index k contains the value of
# the loss function (see below, 𝐿(𝒘, 𝝀)) at the k-1th iteration.
# Implement learning to maximize/minimize the following loss function:
# argmax𝛌 argmin𝐰 𝐿(𝒘, 𝝀)
# 𝐿(𝒘, 𝝀) = 𝒘𝑻𝒘 + ∑ 𝜆𝑖 (1 − (𝒘𝑻𝒙𝒊))𝑖∈+1+ ∑ 𝜆𝑖 (1 + (𝒘𝑻𝒙𝒊)) 𝑖∈−1
# At each iteration, compute w using computeW from question 2 and use the derivative
# update rule
# Δ𝜆𝑖 = 𝜖 (1 − 𝑦𝑖(𝒘𝑻𝒙𝒊))
# to update each 𝜆𝑖.
# b is computed as: 𝑏 = −max𝑖∈+1𝒘𝑇𝒙𝑖+mini∈−1𝒘𝑇𝒙𝑖2
# Implementation notes: initialize w as all 0’s and all data points’ lambdas as 1.
# Set 𝝐 = 𝟎. 𝟎𝟏 as step size.

# this uses the dataSet....looks really complex hopefully not


#  does not work with this... gets to zero loss but seems all wrong
# dataTrain = np.array(
#     [
#         [-3.5, -1, -3, 1, 0.5, 2, 1],
#         # [-1.5, -0.5, -4, -1, 0, 3, 1],
#         [3, 0, 4, 0, 0.5, 0.0, -1],
#     ]
# )

iters = 40

# so w goies yo [1,1] loss goes to zero ... so seems to be working
# the answer is oddly large but is essentially [1,1]
# but the lamtotal is huge, what is going on
dataTrain = np.array(
    [
        [4, 2, 1],
        [2, 4, -1],
    ]
)


# dataTrain = np.genfromtxt("hw2\hw2data.csv", delimiter=",", skip_header=1)

# dataTrain = np.delete(dataTrain, 0, axis=1)

# print(len(dataTrain))
# print(dataTrain[3])


# dataTrain[dataTrain[:, 10] == 0, 10] = -1

# # dataTrain = dataTrain[:100]

# print(dataTrain[3])


def learnLam(dataTrain, iters):

    # need toi know how many features
    n = len(dataTrain[0]) - 1
    # Intialize w to all zeros,
    w = np.zeros(n)
    # lambdas to all 1s
    lams = np.ones(len(dataTrain))
    # print(lams)
    # epsilon to 0.01
    epsilon = 0.01
    # this will hold the change in lambda
    changeLam = np.zeros((len(dataTrain)))

    # splitting data it features and labels
    labels = dataTrain[:][:, -1]
    dataSet = dataTrain[:, 0:-1]

    # our loop for intervals
    for i in range(0, iters):

        # maxmin always starts at zero
        maxmin = 0
        # add lambda total as in theory it should go to zero
        lamTot = 0
        # loopong through the feature
        for j in range(0, len(dataSet)):

            # print(maxmin)
            # this is also formula for change in lambda and argmax lam argmin w
            temp = 1 - labels[j] * (np.dot(w, dataSet[j]))
            # adding them all up
            maxmin += temp

            # getting the change in lambda
            changeLam[j] = epsilon * temp

            # updating our lambda but never going below zero
            if lams[j] - changeLam[j] >= 0:
                lams[j] = lams[j] - changeLam[j]
            else:
                lams[j] = 0

            # I just wanted to keep trak of this
            lamTot += lams[j] * labels[j]
        # addding w tranpose w to maxmin to get loss

        print("lamTot")
        print(lamTot)
        # last part of calculating loss
        loss = np.dot(w, w) + maxmin
        print("loss")
        print(loss)

        print("lambdas")
        print(lams)
        # print(changeLam)

        # compute new w... but when
        w = computeW(lams, labels, dataSet)
        print("w")
        print(w)
        print("--------------------------")

        # for i in range(0, len(lams)):
        #     if lams[i] - changeLam[i] >= 0:
        #         lams[j] = lams[i] - changeLam[i]
        #     else:
        #         lams[j] = 0


learnLam(dataTrain, iters)
