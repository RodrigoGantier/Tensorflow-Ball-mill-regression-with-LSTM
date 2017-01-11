#coding:utf8
'''
the MPL encoding reduces the data set 
'''
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

train_x = np.load('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/tr_x.npy')[:, 550:]
train_y = np.load('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/tr_y.npy')
test_x = np.load('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/te_x.npy')[:, 550:]
test_y = np.load('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/te_y.npy')

train_x = np.append(train_x, train_x, 0)
train_y = np.append(train_y, train_y)
train_x = np.append(train_x, train_x, 0)
train_y = np.append(train_y, train_y)

test_x = np.append(test_x, test_x, 0)
test_y = np.append(test_y, test_y)
test_x = np.append(test_x, test_x, 0)
test_y = np.append(test_y, test_y)

BATCH_START=0
BATCH_START_TEST=0
BATCH_SIZE=250

# Parameters
learning_rate = 0.002
training_epochs = 500000
batch_size = BATCH_SIZE
display_step = 200
KEEP_PROB = 1

# Network Parameters
n_hidden_1 = 400 # 1st layer number of features
n_hidden_2 = 150 # 2nd layer number of features
n_hidden_3 = 250 # 2nd layer number of features
n_input = train_x.shape[1] # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

def get_batch():
    global BATCH_START
    
    # xs shape (50batch, 20steps)
    seq = train_x[BATCH_START:BATCH_START+BATCH_SIZE,:]
    res = train_y[BATCH_START:BATCH_START+BATCH_SIZE].reshape([BATCH_SIZE, n_classes])
    BATCH_START += BATCH_SIZE
    if BATCH_START+BATCH_SIZE>train_x.shape[0]:
        BATCH_START=0
    return seq, res
def get_batch_test():
    global BATCH_START_TEST
    
    # xs shape (50batch, 20steps)
    seq = test_x[BATCH_START_TEST:BATCH_START_TEST+BATCH_SIZE,:]
    res = test_y[BATCH_START_TEST:BATCH_START_TEST+BATCH_SIZE].reshape([BATCH_SIZE, n_classes])
    BATCH_START_TEST += BATCH_SIZE
    if BATCH_START_TEST+BATCH_SIZE>test_x.shape[0]:
        BATCH_START_TEST=0
    return seq, res

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")

# Create model
def multilayer_perceptron(x, weights, biases, keep_prob):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer, layer_3

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred, layer_3 = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
loss = tf.square(tf.sub(pred, y))
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
a=[]
b=[]
c = 100
epoch=0
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    while c>0.1:
        epoch=epoch+1
    #for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = train_x.shape[0]/batch_size
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y, 
                                                          keep_prob: KEEP_PROB})
            # Compute average loss
#             avg_cost += c / total_batch 
        ####################################
            if i%200==0:
                predic = sess.run(pred, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
                error_total = np.float(mean_squared_error(predic, batch_y))
                print('RMSE: ', round(error_total, 2))
                print ('loss: ', c)
        ####################################
            
    print("Optimization Finished!")

    # Test model
    BATCH_START=0
    a=[]
    b=[]
    prediccion = []
    test = []
    total_batch = test_x.shape[0]/batch_size
    for i in range(total_batch):
        batch_x, batch_y = get_batch_test()
        Y_pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        prediccion = np.append(prediccion, Y_pred.flatten())
        test = np.append(test, batch_y.flatten())
        error_total = np.float(mean_squared_error(prediccion, test))
        print('Error total: ', round(error_total, 2))
    train_x = sess.run(layer_3, feed_dict={x: train_x, keep_prob: 1})
    test_x = sess.run(layer_3, feed_dict={x: test_x, keep_prob: 1})
np.save('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/200_v_tr_x.npy',train_x)
np.save('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/MPL_extracion/200_v_te_x.npy', test_x)
