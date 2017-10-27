from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import getWeekday
import numpy as np
import matplotlib.pyplot as plt
import predict_util

def genData(bld_name, n_train, n_valid, n_lag, T, curr_day):
    load_weekday = getWeekday.getWeekdayload(bld_name)
    max_load = np.max(load_weekday)
    min_load = np.min(load_weekday)
    load_weekday = (load_weekday - min_load) / (max_load - min_load)
    
    ################## generate data ##########################################
    y_train = np.zeros((n_train, T))
    X_train = np.zeros((n_train, T * n_lag))
    
    y_valid = np.zeros((n_valid, T))
    X_valid = np.zeros((n_valid, T * n_lag))
    
    row = 0
    for train_day in range(curr_day - n_train - n_valid, curr_day - n_valid):
        y_train[row,:] = load_weekday[train_day * T : train_day * T + T]
        X_train[row,0*T*n_lag:1*T*n_lag] = load_weekday[train_day * T - n_lag * T: train_day * T]
        row += 1
    
    row = 0
    for valid_day in range(curr_day - n_valid, curr_day):
        y_valid[row,:] = load_weekday[valid_day * T : valid_day * T + T]
        X_valid[row,0*T*n_lag:1*T*n_lag] = load_weekday[valid_day * T - n_lag * T: valid_day * T]
        row += 1    
        
    # building test data
    X_test = np.zeros((1, T * n_lag))
    X_test[0, 0*T*n_lag:1*T*n_lag] = load_weekday[curr_day*T - n_lag*T: curr_day*T]
    y_test = load_weekday[curr_day*T: curr_day *T + T]
    
    
    return(X_train, y_train, X_valid, y_valid, X_test, y_test, min_load, max_load)
        
    
def RNN(x, weights, biases, num_hidden, timesteps):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=None)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell for _ in range(2)])
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    
    outputs = tf.reshape(outputs, [-1, timesteps])
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights) + biases




def RNN_LSTM(bld_name, curr_day):    
    # Training Parameters
    training_steps = 10000
    display_step = 100
    
    # Network Parameters
    num_input = 1 # MNIST data input (img shape: 28*28)
    T = 96
    num_hidden = 1 # hidden layer num of features
    n_train = 5
    n_valid = 1
    n_lag = 2
    timesteps = T * n_lag # timesteps
    
    
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, T])
    
    # Define weights
    weights = tf.Variable(tf.random_normal([T*n_lag, T]))
    biases = tf.Variable(tf.random_normal([T]) )
     
    ###############################################################################
    prediction = RNN(X, weights, biases, num_hidden, timesteps)

    
    # Define loss and optimizer
    loss = T * tf.reduce_mean(tf.square(Y - prediction) )  
    #loss += 1e-2 * ( tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases) )
    train_op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)
    
    
    
    (X_train, y_train, X_valid, y_valid, X_test, y_test, min_load, max_load) = genData(bld_name, n_train, n_valid, n_lag, T, curr_day)
    
    '''
    for step in range(training_steps+1):

        X_train = X_train.reshape((n_train, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: X_train, Y: y_train})
        if step > 0 and step % display_step == 0:
            # Calculate batch loss and accuracy
            l = sess.run(loss, feed_dict={X: X_train, Y: y_train})
            print('iteration number %d, loss is %2f' % (step, l))
    '''

    last_loss = 10000.0
    epsilon = 1e-5
    step = 0
    

    
    while(step < training_steps):

        X_train = X_train.reshape((n_train, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: X_train, Y: y_train})
        
        # Calculate loss on validation set
        X_valid = X_valid.reshape((n_valid, timesteps, num_input))
        l = sess.run(loss, feed_dict={X: X_valid, Y: y_valid})
        
        if((step+1) % display_step == 0):
            print('iteration number %d, loss is %2f' % (step+1, l))
        if(abs(last_loss - l) < epsilon ):
            print('training stopped at: iteration number %d, loss is %2f' % (step+1, l))
            break
        else:
            last_loss = l
            step += 1
            

    X_test = X_test.reshape((1, timesteps, num_input))
    y_pred = prediction.eval(session = sess, feed_dict={X: X_test})
    y_pred = y_pred * (max_load - min_load) + min_load
    y_test = y_test * (max_load - min_load) + min_load
    
    mape = predict_util.calMAPE(y_test, y_pred)
    rmspe = predict_util.calRMSPE(y_test, y_pred)
    
    '''
    xaxis = range(T)
    plt.step(xaxis, y_pred.flatten(), 'r')
    plt.step(xaxis, y_test.flatten(), 'g')
    plt.show()  
    '''
    print('MAPE: %.2f, RMSPE: %.2f' % (mape, rmspe))
    
    tf.reset_default_graph()
    sess.close()  
    return (mape, rmspe)

if __name__ == "__main__":
    bld_name = '1008_EE_CSE_WA3_accum'
    load_weekday = getWeekday.getWeekdayload(bld_name)
    T=96
    n_days = int(load_weekday.size / T)
    
    MAPE_sum = 0.0
    RMSPR_sum = 0.0
    
    for curr_day in range(55, n_days-1):
        print(curr_day)
        (mape, rmspe) = RNN_LSTM(bld_name, curr_day)
        MAPE_sum += mape
        RMSPR_sum += rmspe
    
    days_sample = n_days - 1 - 55
    MAPE_sum = MAPE_sum / days_sample
    RMSPR_sum = RMSPR_sum / days_sample
    print('AVERAGE MAPE: %.2f, RMSPE: %.2f' % (MAPE_sum, RMSPR_sum))