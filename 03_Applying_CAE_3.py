
# coding: utf-8

# In[1]:


import tarfile
import pandas as pd
import numpy as np
# from nilearn import plotting
import os
import nibabel as nib
import tensorflow as tf
from tensorflow.python.framework import ops
# import matplotlib.pyplot as plt
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[2]:

os.chdir('/work/05982/tg852820/TeamA')
data_info = pd.read_csv('data_info.csv')


# In[3]:


data_info['Normal'] = data_info["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].apply(
    lambda x: 1 if x == 0 else 0)
data_info['NormalToMCI'] = data_info["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].apply(
    lambda x: 1 if x == 1 else 0)
data_info['MCI'] = data_info["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].apply(
    lambda x: 1 if x == 2 else 0)
data_info['AD'] = data_info["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].apply(
    lambda x: 1 if x == 3 else 0)
data_info['OtherDementia'] = data_info["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].apply(
    lambda x: 1 if x == 4 else 0)


# In[4]:


# data_info.head()


# In[5]:


number_files_loaded = 40
sample_list = data_info.iloc[:number_files_loaded, :]["filename"]

tar = tarfile.open("nacc_copy_TeamA.tar")
for file in sample_list:
    path = "fs_t1/" + file
    tar.extract(path)
    
sample_data_list = list()
for filename in sample_list:
    # or is it better to use get_fdata()?
    a = nib.load("fs_t1/"+filename).get_data()
    sample_data_list.append(a)
sample_dataset = np.array(sample_data_list, dtype=np.float32)
batch_size, height, width, depth = sample_dataset.shape
channels = 1  # gray-scale instead of RGB
s = sample_dataset.reshape(number_files_loaded, 256, 256, 256, 1)

# In[ ]:


def split_train_test(data, test_ratio):
    """
    Generate shuffled indices to split the original dataset.

    Arguments: -- data: dataset to be handled, with shape (n, n_D0, n_H0, n_W0, n_C0) if input is X, (n, n_y) if input is Y
               -- test_ratio: percentage of the test set in the total dataset

    Returns: -- train_indices: a numpy array of the indices to be chosen for the training set 
                               with size len(data)-int(len(data)*test_ratio)
             -- test_indices: a numpy array of the indices to be chosen for the test set
                              with size int(len(data)*test_ratio)
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return train_indices, test_indices


# In[ ]:


train_indices, test_indices = split_train_test(s, 0.1)
X_train_orig = s[train_indices]
X_test_orig = s[test_indices]

X_train = X_train_orig/255.
Y_train = np.clip(X_train, 0., 1.)
X_test = X_test_orig/255.
Y_test = np.clip(X_test, 0., 1.)

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_train shape: " + str(Y_train.shape))
print("Y_test shape: " + str(Y_test.shape))


# In[6]:


def create_placeholders(n_D0, n_H0, n_W0, n_C0):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_D0 -- scalar, depth of an input image
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input

    Returns:
    X -- placeholder for the data input, of shape [None, n_D0, n_H0, n_W0, n_C0] and dtype "float"
    """

    X = tf.placeholder(shape=(None, n_D0, n_H0, n_W0, n_C0), dtype=tf.float32)

    return X


# In[ ]:


def forward_propagation(X, parameters=None):
    """
    Implements the forward propagation for the model:
    CONV3D -> CONV3D -> MAXPOOL -> CONV3D -> MAXPOOL -> CONV3D -> MAXPOOL -> (encoder part)
    UNPOOL -> DECONV3D -> UNPOOL -> DECONV3D -> UNPOOL -> DECONV3D -> DECONV3D (decoder part)

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)

    Returns:
    P62 -- the output of the last layer, which has the same shape as the input X
    """

    # ENCODER

    # CONV3D: number of filters in total 16, stride 1, padding 'SAME', activation 'relu', kernel parameter initializer 'xavier'
    # output_size = (3, 256, 256, 256, 16)
    A11 = tf.layers.conv3d(inputs=X, filters=4, kernel_size=3, padding="SAME", strides=1,
                           activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # CONV3D: number of filters in total 16, stride 1, padding 'SAME', activation 'relu', kernel parameter initializer 'xavier'
    # output_size = (3, 256, 256, 256, 16)
    A12 = tf.layers.conv3d(inputs=X, filters=8, kernel_size=3, padding="SAME", strides=1,
                           activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # MAXPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 128, 128, 128, 16)
    P1 = tf.layers.max_pooling3d(A12, pool_size=3, strides=2, padding="SAME")

    # CONV3D: number of filters in total 32, stride 1, padding 'SAME', activation 'relu', kernel parameter initializer 'xavier'
    # output_size = (3, 128, 128, 128, 32)
    A2 = tf.layers.conv3d(inputs=P1, filters=16, kernel_size=3, padding="SAME", strides=1,
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # MAXPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 64, 64, 64, 32)
    P2 = tf.layers.max_pooling3d(A2, pool_size=3, strides=2, padding="SAME")

    # CONV3D: number of filters in total 32, stride 1, padding 'SAME', activation 'relu', kernel parameter initializer 'xavier'
    # output_size = (3, 64, 64, 64, 32)
    A3 = tf.layers.conv3d(inputs=P2, filters=32, kernel_size=3, padding="SAME", strides=1,
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # MAXPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 32, 32, 32, 32)
    P3 = tf.layers.max_pooling3d(A3, pool_size=3, strides=2, padding="SAME")

    # DECODER

    # UNPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 64, 64, 64, 32)
    A4 = tf.keras.backend.resize_volumes(P3, 2, 2, 2, "channels_last")
    # DECONV3D: number of filters in total 16, stride 1, padding 'SAME', activation 'relu'
    # output_size = (3, 64, 64, 64, 32)
    P4 = tf.layers.conv3d_transpose(
        inputs=A4, filters=16, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)

    # UNPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 128, 128, 128, 32)
    A5 = tf.keras.backend.resize_volumes(P4, 2, 2, 2, "channels_last")
    # DECONV3D: number of filters in total 16, stride 1, padding 'SAME', activation 'relu'
    # output_size = (3, 128, 128, 128, 16)
    P5 = tf.layers.conv3d_transpose(
        inputs=A5, filters=8, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)

    # UNPOOL: window 3x3, sride 2, padding 'SAME'
    # output_size = (3, 256, 256, 256, 16)
    A6 = tf.keras.backend.resize_volumes(P5, 2, 2, 2, "channels_last")
    # DECONV3D: number of filters in total 1, stride 1, padding 'SAME', activation 'relu'
    # output_size = (3, 256, 256, 256, 1)
    P61 = tf.layers.conv3d_transpose(
        inputs=A6, filters=4, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)
    # DECONV3D: number of filters in total 1, stride 1, padding 'SAME', activation 'relu'
    # output_size = (3, 256, 256, 256, 1)
    P62 = tf.layers.conv3d_transpose(
        inputs=P61, filters=1, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)

    return P62


# In[ ]:


def compute_cost(P62, Y):
    """
    Computes the cost

    Arguments:
    P62 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, n_y)
    X -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    logits = tf.nn.sigmoid(P62)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
    cost = tf.reduce_mean(cost)

    return cost


# with tf.Session() as sess:
#     X = create_placeholders(256, 256, 256, 1)
#     Y = create_placeholders(256, 256, 256, 1)
#     P62 = forward_propagation(X)
#     init_op = tf.global_variables_initializer()
#     cost = compute_cost(P62, Y)
#     sess.run(init_op)
#     cost_ = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
# #     decode_output = sess.run(P62, feed_dict={X: s})
# #     encode_output = sess.run(P3, feed_dict={X: s})
# print(cost_)


# In[ ]:


def random_mini_batches(X, Y, mini_batch_size=1, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, n_D0, n_H0, n_W0, n_C0)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    # To make your "random" minibatches the same as ours
    np.random.seed(seed)
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



# minibatch = random_mini_batches(X_train, Y_train)[0]
# mini_X, mini_Y = minibatch

# with tf.Session() as sess:
#     X = create_placeholders(256, 256, 256, 1)
#     Y = create_placeholders(256, 256, 256, 1)
#     P62 = forward_propagation(X)
#     init_op = tf.global_variables_initializer()
#     minibatch = random_mini_batches(X, Y)
#     sess.run(init_op)
#     batch = sess.run(minibatch, feed_dict={X: X_train, Y: Y_train})
#     decode_output = sess.run(P62, feed_dict={X: s})
#     encode_output = sess.run(P3, feed_dict={X: s})
# print(mini_X.shape)


# In[ ]:


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=3, minibatch_size=1, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 256 256, 256, 1)
    Y_train -- test set, of shape (None, n_y = 5)
    X_test -- training set, of shape (None, 256, 256, 256, 1)
    Y_test -- test set, of shape (None, n_y = 5)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()
    # to keep results consistent (tensorflow seed)
    tf.set_random_seed(1)
    # to keep results consistent (numpy seed)
    seed = 3
    (m, n_D0, n_H0, n_W0, n_C0) = X_train.shape
    # To keep track of the cost
    costs = []

    # Create Placeholders of the correct shape
    X = create_placeholders(n_D0, n_H0, n_W0, n_C0)
    Y = create_placeholders(n_D0, n_H0, n_W0, n_C0)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    P62 = forward_propagation(X)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(P62, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(
                X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Run the session to execute the optimizer and the cost
                # The feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={
                                        X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(P62, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy


# In[ ]:


_, _ = model(X_train, Y_train, X_test, Y_test, learning_rate=0.006, num_epochs=5, minibatch_size=6)

