# 4. Presume we have a np array Data with 5000 data points (as rows) and 10
# features (as columns). A final column provides the class label of class 0 or 1.
# ( numpy: Data.shape is [5000, 11] ).
# Define the function meanFeature that takes in the Data matrix and a
# specified classVal (y value 0 or 1) and finds the mean value for each
# feature for that classVal as a 10-element numpy array.
# meanFeature(Data,classVal) would return an np-array like:
# np.array([5, -10, 3.5, -2.5, 0, 12, -13.5, 42, 12.2, -10])

# ok so we need a function
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

size = 5

# Create a 100x11 array with random integers
data = np.random.randint(0, 100, (size, 11))

# Set the last column to 0 or 1 randomly
data[:, -1] = np.random.randint(0, 2, size)

print(data)


def meanFeature(data, classVal):

    print("---------------------------")

    # so first we would need to get just the rows with the correct class values
    # well that works would probably never remeber it
    # so data[:, 10] == seems to let me take an np array and choose rows by column idx 10 value
    mask = data[:, 10] == classVal
    # print(mask)
    # so mask is then a row of true false
    # and then data[mask, :] seems to say only keep the rows that had true.. so [rows, columns]
    data = data[mask, :]

    print(data)

    print("---------------------------")

    # then we drop the last column
    # mp.delete seems easy enought
    # data is our actual array
    # 10 is my column
    # axis = 1 seems to say don't mess with the rows just delete the column, ohh proably beacuse
    # axis one means we care about the columns, so it goes ahead and deletes column 11, if axis = 0 it
    # would try and delete row 10
    data = np.delete(data, 10, axis=1)

    print(data)

    print("---------------------------")

    # now we need the mean

    # so np has a mean function, we send it an np array and then axis = 0 means we want it to
    # get the means of column over all rows while axis = 1 would mean to get the mean of each row... seems
    # a bit counter intuative but let me think about it for a moment. I guiess it is like get the mean by each row
    # or by each column which makes a bit more sense
    return np.mean(data, axis=0)

    # so that seems to work, would I remember how to do any of that probably not


# how professor did it I don't like this
# def meanFeature(Data, classVal):
#     # so this is just a single row array of 10 columns
#     outMeans = np.zeros(Data.shape[1])

#     print("---------------------------")

#     print(outMeans)

#     # so that seems to be the number of rows
#     # print(Data.shape[1])

#     for i in range(Data.shape[1]):

#         inds = np.where(Data[:, -1] == classVal)[0]
#         print(inds)
#         print("---------------------------")

#         outMeans[i] = Data[inds, i].mean()
#         print(outMeans)
#         print("---------------------------")
#     return outMeans


print(meanFeature(data, 1))
