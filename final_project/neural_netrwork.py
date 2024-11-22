import math
import random
import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


# an easy input

# XOR

input_1 = [0, 0, 0]

input_2 = [0, 1, 1]

input_3 = [1, 0, 1]

input_4 = [1, 1, 0]


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


# seperating features and classes

inputs = np.array([input_1, input_2, input_3, input_4])

print(inputs)

classes = inputs[:, 0]

print(classes)

features = np.delete(inputs, 0, axis=1)

print(features)

# maybe doing one bias per nueron just like math for data science waiting to hear back will start that way
# does not seem like the best way to do this but also seems slightly better than original version
layers = 2

layer_1 = 2

layer_2 = 1

epsil = 0.1
# want a np array of this size

# weights layer 1, each layer represents the weights to one neuron
# wl1 = np.random.uniform(-1, 1, size=(layer_1, features))
# wl1 = np.random.randint(1, 10, size=(layer_1, features))
wl1 = np.array([[0.4, 0.8], [-0.4, -0.4]])

print(wl1)

# weights layer 2, each layer represents the weights to one neuron
# wl2 = np.random.uniform(-1, 1, size=(layer_2, layer_1))
# wl2 = np.random.randint(1, 10, size=(layer_2, features))
wl2 = np.array([[0.2, 0.3]])


# in theory i can put the wights into somthing like this, in practice I will rpobably not
# test1 = np.array([wl1, wl2], dtype=object)

# print(test1)


# for i in range(len(test1)):
#     for j in range(len(test1[i])):
#         print(test1[i][j])

# print("wl2")
# print(wl2)


# bias weights layer 1
# bwl1 = np.random.uniform(-1, 1, layer_1)
bwl1 = np.array([5, 6])

# bias weights layer 2
# bwl2 = np.random.uniform(-1, 1, layer_2)
bwl2 = np.array([0.2])

bias_weights = np.array([bwl1, bwl2], dtype=object)

# neurons layer 1, start at zero
nl1 = np.zeros(layer_1)

# neurons layer 2, start at zero
nl2 = np.zeros(layer_2)

neurons = np.array([nl1, nl2], dtype=object)

# for i in range(len(neurons)):
#     print(neurons[i])
#     print("----")
#     for j in range(len(neurons[i])):
#         print(neurons[i][j])

# print(neurons)

# so ideally my neurons would all be in one array and i would loop through the array to get all the levels
# this is better than what I had but it could be better
###################################################################
#     forward propagation layer 1
##################################################################


# wights times features level 1
wtfl1 = wl1 * [2, 1]

# sum of wights times features level 1
swtfl1 = np.sum(wtfl1, axis=1)
print(swtfl1)

# sum of weight + biases
total_sum = bias_weights[0] + swtfl1
print(total_sum)

# sending total sums to sigmoid function
for i in range(len(total_sum)):
    # that will be the value of our layer 1 neurons
    neurons[0][i] = sig(total_sum[i])

print("neurons[0]")
print(neurons[0])


###################################################################
#     forward propagation layer 2
##################################################################


# layer 2 weights times neurons of layer 1
lw2n1 = neurons[0] * wl2

print(lw2n1)

# sum of wights times features level 1
swtfl2 = np.sum(lw2n1, axis=1)
print(swtfl2)

# sum of weight + biases
total_sum_2 = bias_weights[1] + swtfl2
print(total_sum_2)

for i in range(len(total_sum_2)):
    neurons[1][i] = sig(total_sum_2[i])

print("neurons[1]")
print(neurons[1])

# calculate error??

###################################################################
#     backward propagation layer 2
##################################################################

# step one calculate delta

delta_top = (1 - neurons[1]) * (1 - neurons[1]) * neurons[1]

print("delta_top")
print(delta_top)

# just the rest of the formula with the handy dandy use of arrays
wl2 = wl2 + (epsil * delta_top * wl2)

# probably a better way to combine these but this works for getting bias weights at least if that 1 value
# is correct
bias_weights[1] = bias_weights[1] + (epsil * delta_top)

print("layer 2 updated weights regular and bias")
print(wl2)
print(bias_weights[1])


###################################################################
#     backward propagation layer 1
##################################################################


# so here it gets a bit harder we have the delta

# but this does seem to get us our new error correction thing
error_corr = np.sum(delta_top * wl1, axis=1)
print(error_corr)

# I think this is the lower cas delta
lil_deltal1 = (1 - neurons[0]) * error_corr * neurons[0]

print(lil_deltal1)

print("changed weights layer 1")
print(wl1 + epsil * lil_deltal1 * [2, 1])

# # eta

# eta = 0.1

# # need to start the error to get us into the while loop
# error = 100

# counter = 0

# # an array for our errors
# all_errors = []


# while error > 0.0005:

#     # reset the error to zero as we will calculate it each time
#     error = 0

#     # shuffles our inputs around, I think this is helpful but not entirely sure
#     random.shuffle(inputs)

#     counter += 1

#     for i in inputs:


# checking final neuron vs expected value
#         # ERROR
#         error = error + 0.5 * pow((t_1 - y1), 2)

#         # print(f'error = {error}\n')

#         # BACKWARD PROPOGATION

#         # DELTA K
#         # so delta k will change the weights of all the hideen layer weights
#         delta_k1 = y1 * (1 - y1) * (t_1 - y1)

#         # print(f'delta_k1 = {delta_k1}\n')

#         # the delta_js change the input layer weights
#         # delta_j1 for the weights going to j1 (1,1) (2,1) + the bias weight
#         delta_j1 = zz1 * (1 - zz1) * (wjk_11 * delta_k1)

#         # print(f'delta_j1 = {delta_j1}\n')

#         # deltaj2 for those going to j2 (1,2) (2,2) and the bias weight
#         delta_j2 = zz2 * (1 - zz2) * (wjk_21 * delta_k1)

#         # print(f'delta_j2 = {delta_j2}\n')

#         # now we calculate the changes to the weights and update them
#         delta_wjk_11 = eta * zz1 * delta_k1

#         # print(f'delta_wjk_11 = {delta_wjk_11}\n')

#         delta_wjk_21 = eta * zz2 * delta_k1
#         # print(f'delta_wjk_21 = {delta_wjk_21}\n')

#         # pretty sure this is the right formula, as we don't seem to to be using the weights in the pervious 2 formulas
#         delta_wbb1 = eta * bb1 * delta_k1

#         # print(f'delta_wbb1 = {delta_wbb1}\n')

#         # Update the weights
#         wjk_11 += delta_wjk_11

#         # print(f'wjk_11 = {wjk_11}\n')

#         wjk_21 += delta_wjk_21

#         # print(f'wjk_21 = {wjk_21}\n')

#         wbb1 += delta_wbb1

#         # print(f'wbb1 = {wbb1}\n')

#         # calvulating delta for j1 weights

#         delta_wij_11 = eta * z1 * delta_j1

#         # print(f'delta_wij_11 = {delta_wij_11}\n')

#         delta_wij_21 = eta * z2 * delta_j1

#         # print(f'delta_wij_21 = {delta_wij_21}\n')

#         # pretty sure this is the right formula, as we don't seem to to be using the weights in the pervious 2 formulas
#         delta_wb1 = eta * b1 * delta_j1

#         # print(f'delta_wb1 = {delta_wb1}\n')

#         # Update the weights for j1
#         wij_11 += delta_wij_11

#         # print(f'wij_11 = {wij_11}\n')

#         wij_21 += delta_wij_21

#         # print(f'wij_21 = {wij_21}\n')

#         wb1 += delta_wb1

#         # print(f'wb1 = {wb1}\n')

#         # calvulating delta for j2 weights

#         delta_wij_12 = eta * z1 * delta_j2

#         # print(f'delta_wij_12 = {delta_wij_12}\n')

#         delta_wij_22 = eta * z2 * delta_j2

#         # print(f'delta_wij_22 = {delta_wij_22}\n')

#         # pretty sure this is the right formula, as we don't seem to to be using the weights in the pervious 2 formulas
#         delta_wb2 = eta * b2 * delta_j2

#         # print(f'delta_wb2 = {delta_wb2}\n')

#         # Update the weights for j2
#         wij_12 += delta_wij_12

#         # print(f'wij_12 = {wij_12}\n')

#         wij_22 += delta_wij_22

#         # print(f'wij_22 = {wij_22}\n')

#         wb2 += delta_wb2

#         # print(f'wb2 = {wb2}\n')

#     all_errors.append(error)

# # print(f't1 = {t_1} & y1 = {y1}\n')

# print(f"error = {error}\n")


# print(f"counter = {counter}\n")


# print(f"wij_11 = {wij_11}\n")

# print(f"wij_12 = {wij_12}\n")

# print(f"wij_21 = {wij_21}\n")

# print(f"wij_22 = {wij_22}\n")

# print(f"wjk_11 = {wjk_11}\n")

# print(f"wjk_21 = {wjk_21}\n")

# print(f"wb1 = {wb1}\n")

# print(f"wb2 = {wb2}\n")

# print(f"wbb1 = {wbb1}\n")

# # running one last time with the correct weights so I can get the final results for all the inputs

# error = 0


# for i in inputs:

#     z1 = i[0]

#     z2 = i[1]

#     t_1 = i[2]

#     net_output_1 = b1 * wb1 + z1 * wij_11 + z2 * wij_21

#     net_output_2 = b2 * wb2 + z1 * wij_12 + z2 * wij_22

#     zz1 = 1 / (1 + math.exp(-net_output_1))
#     zz2 = 1 / (1 + math.exp(-net_output_2))

#     net_output_final = bb1 * wbb1 + zz1 * wjk_11 + zz2 * wjk_21

#     y1 = 1 / (1 + math.exp(-net_output_final))

#     #  seems to be working :)
#     print(f"t1 = {t_1} | x1 = {z1} | x2 = {z2} | y1 = {y1}\n")

#     # ERROR
#     error = error + 0.5 * pow((t_1 - y1), 2)


# print(f"error = {error}\n")


# # if you want to see more of the errors just use this
# # for i in range(len(all_errors)):
# #     if i % 1000 == 0:
# #         print(all_errors[i])
