#coding:utf8
'''
Created on Jan 9, 2017

@author: lab
'''

import tensorflow as tf
import numpy as np
    
train_x = np.load("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_train_x.npy")
train_y = np.load("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_train_y.npy")
test_x = np.load("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_test_x.npy")
test_y = np.load("/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/550_r_test_y.npy")

train_x =train_x[:, 7, :, :].reshape([-1,550]) 
test_x = test_x[:, 7, :, :].reshape([-1,550])

tr_x = train_x
te_x = test_x
    
#duplicate data
tr_x = np.append(tr_x, tr_x, 0)
tr_x = np.append(tr_x, tr_x, 0)
te_x = np.append(te_x, te_x, 0)
te_x = np.append(te_x, te_x, 0)

n_visible = 550
n_hidden_1 = 300
n_hidden_2 = 200
corruption_level = 0.3
batch_size = 350
KP = 1.0
corruption_level = 0.1

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')
keep_prob = tf.placeholder("float")
# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden_1))
W_init = tf.random_uniform(shape=[n_visible, n_hidden_1], minval=-W_init_max, maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden_1]), name='b')

W_init_max = 4 * np.sqrt(6. / (n_hidden_1 + n_hidden_2))
W_init = tf.random_uniform(shape=[n_hidden_1, n_hidden_2], minval=-W_init_max, maxval=W_init_max)

W1 = tf.Variable(W_init, name='W')
b1 = tf.Variable(tf.zeros([n_hidden_2]), name='b')

W_prime = tf.transpose(W1)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_hidden_1]), name='b_prime')

W_prime1 = tf.transpose(W)  # tied weights between encoder and decoder
b_prime1 = tf.Variable(tf.zeros([n_visible]), name='b_prime')


def model(X, mask, W, b, W_prime, b_prime, W1, b1, W_prime1, b_prime1, keep_prob):
    
    tilde_X = mask * X # corrupted X
    h1 = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    h1 = tf.nn.dropout(h1, keep_prob)
    h2 = tf.nn.sigmoid(tf.matmul(h1, W1) + b1)  # reconstructed input
    h2 = tf.nn.dropout(h2, keep_prob)
    h3 = tf.nn.sigmoid(tf.matmul(h2,W_prime) + b_prime)
    h3 = tf.nn.dropout(h3, keep_prob)
    h4 = tf.nn.sigmoid(tf.matmul(h3, W_prime1) + b_prime1)
    
    return h4, h2

# build model graph
Z, encoder = model(X, mask, W, b, W_prime, b_prime, W1, b1, W_prime1, b_prime1, keep_prob)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2))  # minimize squared error
train_op = tf.train.AdamOptimizer(0.002).minimize(cost)  # construct an optimizer

# Launch the graph in a session
c = 100
epoch = 0
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    while c>0.05:
        epoch = epoch+1
    #for i in range(5000):
        for start, end in zip(range(0, len(tr_x), batch_size), range(batch_size, len(tr_x), batch_size)):
            input_ = tr_x[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np, keep_prob: KP})
        if epoch%5==0:
            mask_np = np.random.binomial(1, 1 - 0, test_x.shape)
            c = sess.run(cost, feed_dict={X: test_x, mask: mask_np, keep_prob: 1})
            print c
    mask_np = np.random.binomial(1, 1 - 0, train_x.shape)
    train_x = sess.run(encoder,feed_dict={X: train_x, mask: mask_np, keep_prob: 1})
    mask_np = np.random.binomial(1, 1 - 0, test_x.shape)
    test_x = sess.run(encoder,feed_dict={X: test_x, mask: mask_np, keep_prob: 1})
    np.save('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/200_enc_v_train_x.npy', train_x)
    np.save('/media/lab/办公/extract data Rodrigo/Rodrigo/laoshi_data/extraccion/200_enc_v_test_x.npy', test_x)
