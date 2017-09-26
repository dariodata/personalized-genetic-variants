import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

np.random.seed(1)

# load the data
X_train_orig = np.load('input/train_alt.npy')
Y_train_orig = np.load('input/encoded_y.npy')
X_test = np.load('input/test_alt.npy')
test_index = np.load('input/test_index.npy')
timestr = time.strftime("%Y%m%d-%H%M%S")
dirname = 'output/'  # output directory
filename = ''

X_train, X_val, Y_train, Y_val = train_test_split(X_train_orig, Y_train_orig, test_size=0.18, random_state=42)
X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T
X_test = X_test.T
print('X_train: ', X_train.shape)
print('X_val: ', X_val.shape)
print('Y_train: ', Y_train.shape)
print('Y_val: ', Y_val.shape)


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, dimensions of the input
    n_y -- scalar, number of classes (from 0 to 8, so -> 9)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow.

    Returns:
    parameters -- a dictionary of tensors containing W and b for every layer
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [350, X_train.shape[0]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [350, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [350, 350], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [350, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [100, 350], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [100, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable('W4', [9, 100], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable('b4', [9, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}

    return parameters


def forward_propagation(X, parameters, keep_prob1, keep_prob2):
    """
    Implements the forward propagation for the model: (LINEAR -> RELU)^3 -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W" and "b" for every layer
                  the shapes are given in initialize_parameters

    Returns:
    Z4 -- the output of the last LINEAR unit (logits)
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.matmul(W1, X) + b1  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    A1 = tf.nn.dropout(A1, keep_prob1) # add dropout
    Z2 = tf.matmul(W2, A1) + b2  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    A2 = tf.nn.dropout(A2, keep_prob2) # add dropout
    Z3 = tf.matmul(W3, A2) + b3  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4, A3) + b4

    return Z4


def compute_cost(Z4, Y):
    """
    Computes the cost

    Arguments:
    Z4 -- output of forward propagation (output of the last LINEAR unit), of shape (n_classes, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z4

    Returns:
    cost - Tensor of the cost function
    """

    # transpose to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def random_mini_batches(X, Y, mini_batch_size, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector, of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters['W1'])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4}

    x = tf.placeholder("float", [X_train.shape[0], None])
    keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
    keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')

    z4 = forward_propagation(x, params, keep_prob1, keep_prob2)
    p = tf.nn.softmax(z4, dim=0) # dim=0 because the classes are on that axis
    # p = tf.argmax(z4) # this gives only the predicted class as output

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X, keep_prob1:1.0, keep_prob2:1.0})

    return prediction


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1000, minibatch_size=64, print_cost=True):
    """
    Implements a four-layer tensorflow neural network: (LINEAR->RELU)^3->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size, number of training examples)
    Y_train -- test set, of shape (output size, number of training examples)
    X_test -- training set, of shape (input size, number of training examples)
    Y_test -- test set, of shape (output size, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost
    t0 = time.time()  # to mark the start of the training

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')  # probability to keep a unit during dropout
    keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation
    Z4 = forward_propagation(X, parameters, keep_prob1, keep_prob2)

    # Cost function
    cost = compute_cost(Z4, Y)
    regularizers = tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2']) + tf.nn.l2_loss(parameters['W3']) \
                   + tf.nn.l2_loss(parameters['W4']) # add regularization term
    beta = 0.1 # regularization constant
    cost = tf.reduce_mean(cost + beta*regularizers) # cost with regularization

    # Backpropagation: Define the tensorflow AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost"
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y,
                                                                           keep_prob1: 0.7, keep_prob2: 0.5})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch {}: {:f}".format(epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_cost = cost.eval({X: X_train, Y: Y_train, keep_prob1:1.0, keep_prob2:1.0})
        test_cost = cost.eval({X: X_test, Y: Y_test, keep_prob1:1.0, keep_prob2:1.0})
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob1:1.0, keep_prob2:1.0})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob1:1.0, keep_prob2:1.0})

        print('Finished training in %s s' % (time.time() - t0))
        print("Train Cost:", train_cost)
        print("Test Cost:", test_cost)
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate = {}, beta = {},\n"
                  "test cost = {:.6f}, test accuracy = {:.6f}".format(learning_rate, beta, test_cost, test_accuracy))
        global filename
        filename = timestr + '_NN4L_alt_lr_{}_beta_{}_cost_{:.2f}-{:.2f}_acc_{:.2f}-{:.2f}'.format(
            learning_rate, beta, train_cost, test_cost, train_accuracy, test_accuracy)
        plt.savefig(dirname + filename + '.png')

        return parameters


parameters = model(X_train, Y_train, X_val, Y_val)

prediction = predict(X_test, parameters)
submission = pd.DataFrame(prediction.T)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv(dirname + filename + '.csv', index=False)
