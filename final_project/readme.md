Hello! And welcome to my very simple neural network

There is one main function neural_net(\_epochs, \_epsil, \_features, \_classes, \_layers, \_supress_threshold, \_k_iter,
):

It takes in the number of epochs you want to run (integer), the learning rate (epsilon) to use (real number), your data dived into features and classes, the number of hidden layers and size of each hidden layer to use (as an np array of the size you want). The final layer does need to have only one neuron so only two classes can be dealt with. I feel confident I could change this but I have not changed it so that is where it stands at the moment. The threshold at which to suppress a connection (supress_threshold, any real number) and the number of epoch at which to check for suppression (k_iter, an integer)

It will then return the weights (regular and bias) an np array of the neurons, this is used for checking testing how we did and there is probably a better way to do it but it is what I have at the moment, and an array of errors, also to help check how we did.

There are a two helper functions, a very simple sigmoid function (sig) and a pruning function pruning(original, prune, supress_threshold, iter, k_iter) that suppresses weights when they fall below a suppression threshold (supress_threshold) and it will check at a specified amount of epochs (k_iters). This is placed within the larger neural_net function so users really don't need to worry about it

If you want to change the data first go to the section marked Input any changes that need to be done.

YOur classes need to be in the np array named classes and same for the features, this data has the classes in the first column (0) but if yours is different just make the appropriate changes.

Then to the section marked checking test data, same as above

All the the parameters and hyperparameters can be changed in the section marked USER INPUT, they are hopefully self explanatory.
Once you run the program you will be given a graph of the error rate over the epochs and then a confusion matrix from your training data and then test data.

I hope you find it useful
