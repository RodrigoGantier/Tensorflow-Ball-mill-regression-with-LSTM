#coding:utf8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#from laoshi data
train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")

#duplicate data
train_x = np.append(train_x, train_x, 0)
train_y = np.append(train_y, train_y)
train_x = np.append(train_x, train_x, 0)
train_y = np.append(train_y, train_y)

test_x = np.append(test_x, test_x, 0)
test_y = np.append(test_y, test_y)
test_x = np.append(test_x, test_x, 0)
test_y = np.append(test_y, test_y)

# Network Parameters
n_input = train_x.shape[1] # laoshi data input (img shape: 1*240)
n_steps = 6 # timesteps
n_hidden = 250 # hidden layer num of features
h1=250        #lstm units
n_output = 1 #total output (0-9 digits)
learning_rate = 0.0015
batch_init = 0
training_iters = 22440
batch_size = 250
ventana = 11

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder("float")

w_fc1 = tf.Variable(tf.random_normal([n_input, h1])) # Hidden layer weights
b_fc1 = tf.Variable(tf.random_normal([h1]))

w_fc2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
b_fc2 = tf.Variable(tf.random_normal([n_output]))

x_internal = tf.transpose(x, [1, 0, 2])
x_internal = tf.reshape(x_internal, [-1, n_input]) # (n_steps*batch_size, n_input)
            
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
istate = lstm_cell.zero_state(batch_size*(ventana-n_steps), tf.float32)

h_fc1 = tf.nn.relu(tf.matmul(x_internal, w_fc1) + b_fc1)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
h_fc1 = tf.split(0, n_steps, h_fc1) # n_steps * (batch_size, n_hidden)

outputs, states = tf.nn.rnn(lstm_cell, h_fc1, dtype=tf.float32)
y = tf.add(tf.matmul(outputs[-1], w_fc2), b_fc2)

mse = tf.reduce_mean(tf.square(y_-y)) # MSE loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)


def sequencesFromTrainingData(batch_size, n_steps, n_input, n_output): 
    ''' This function creates several training sequences from each
         MNIST sample image, each training sequence consists of #n_steps samples
         with 28 features.
    '''
    global batch_init
    X = []
    Y = []
    
    data_x = np.split(train_x[batch_init:batch_init+ventana*batch_size,:], batch_size, axis=0)
    data_y = np.split(train_y[batch_init:batch_init+ventana*batch_size], batch_size, axis=0)
    for p in range(batch_size):
        sample = data_x[p]
        lavels = data_y[p]
        for i in range(ventana-n_steps):
            X.append(sample[i:i+n_steps,:])
            Y.append(lavels[i+n_steps])
    
    batch_init = batch_init + ventana*batch_size  
    if (batch_init+ventana*batch_size)>=train_x.shape[0]:
        batch_init = 0
    
    return np.asarray(X), np.asarray(Y)[:, np.newaxis]
        
def sequencesFromTestData(n_steps): 
    ''' This function creates several training sequences from each
         MNIST sample image, each training sequence consists of #n_steps samples
         with 28 features.
    '''
    global batch_init
    X = []
    Y = []
    batch_size = test_x.shape[0]/ventana
    data_x = np.split(test_x, batch_size, axis=0)
    data_y = np.split(test_y, batch_size, axis=0)
    for p in range(batch_size):
        sample = data_x[p]
        lavels = data_y[p]
        for i in range(ventana-n_steps):
            X.append(sample[i:i+n_steps,:])
            Y.append(lavels[i+n_steps])
    
    return np.asarray(X), np.asarray(Y)[:, np.newaxis]

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

loss = 1000
step = 1

#while loss > 0.01:
for i in range(training_iters):
    
    batch_xs, batch_ys = sequencesFromTrainingData(batch_size, n_steps, n_input, n_output)
    sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})

    if step % 200 == 0:
        prediccion, loss = sess.run([y, mse], feed_dict={x: batch_xs, y_: batch_ys, keep_prob:1})
        print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss)
        #plt.plot(batch_ys.flatten(), 'r', prediccion.flatten(), 'b--')
   
    step += 1
    
print "Optimization Finished!" 
batch_init = 0
predic = np.array([])
task = np.array([])

for pp in range(2):
    plt.ion()
    plt.show()
    batch_xs, batch_ys = sequencesFromTestData(n_steps)
    for i in range(batch_xs.shape[0]):
        p = sess.run(y, feed_dict={x: batch_xs[i,:,:][np.newaxis, :, :], istate: [np.zeros((batch_size*(ventana-n_steps), n_hidden)), np.zeros((batch_size*(ventana-n_steps), n_hidden))], keep_prob:1} )
  
        predic = np.append(predic, p[0,0])
        task   = np.append(task, batch_ys[i, :][0])

        error = np.sqrt(np.square(predic-task).mean())
        print 'RMS-E: %f'%error
    plt.plot(task, 'r', predic, 'b--')
    print 'RMS-E error final: %f'%error
