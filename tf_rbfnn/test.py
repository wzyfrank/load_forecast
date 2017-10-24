# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:09:31 2016

@author: Rob Romijnders
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import matplotlib.pyplot as plt

#Load the MNIST data
#X_train and y_train refers to the usual 60.000 by 784 matrix and 60.000 vector
#X_test and y_test refers to the usual 10.000 by 784 and 10.000 vector
X_test = np.loadtxt('/home/rob/Dropbox/ml_projects/RCNN/MNIST_data/X_test.csv', delimiter=',')
y_test = np.loadtxt('/home/rob/Dropbox/ml_projects/RCNN/MNIST_data/y_test.csv', delimiter=',')
X_train = np.loadtxt('/home/rob/Dropbox/ml_projects/RCNN/MNIST_data/X_train.csv', delimiter=',')
y_train = np.loadtxt('/home/rob/Dropbox/ml_projects/RCNN/MNIST_data/y_train.csv', delimiter=',')

"""Hyper-parameters"""
batch_size = 300            # Batch size for stochastic gradient descent
test_size = batch_size      # Temporary heuristic. In future we'd like to decouple testing from batching
num_centr = 150             # Number of "hidden neurons" that is number of centroids
max_iterations = 1000       # Max number of iterations
learning_rate = 5e-2        # Learning rate
num_classes = 10            # Number of target classes, 10 for MNIST
var_rbf = 225               # What variance do you expect workable for the RBF?

#Obtain and proclaim sizes
N,D = X_train.shape         
Ntest = X_test.shape[0]
print('We have %s observations with %s dimensions'%(N,D))

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

#Placeholders for data
x = tf.placeholder('float',shape=[batch_size,D],name='input_data')
y_ = tf.placeholder(tf.int64, shape=[batch_size], name = 'Ground_truth')


with tf.name_scope("Hidden_layer") as scope:
  #Centroids and var are the main trainable parameters of the first layer
  centroids = tf.Variable(tf.random_uniform([num_centr,D],dtype=tf.float32),name='centroids')
  var = tf.Variable(tf.truncated_normal([num_centr],mean=var_rbf,stddev=5,dtype=tf.float32),name='RBF_variance')
  
  #For now, we collect the distanc
  exp_list = []
  for i in xrange(num_centr):
        exp_list.append(tf.exp((-1*tf.reduce_sum(tf.square(tf.sub(x,centroids[i,:])),1))/(2*var[i])))
        phi = tf.transpose(tf.pack(exp_list))
        
with tf.name_scope("Output_layer") as scope:
    w = tf.Variable(tf.truncated_normal([num_centr,num_classes], stddev=0.1, dtype=tf.float32),name='weight')
    bias = tf.Variable( tf.constant(0.1, shape=[num_classes]),name='bias')
        
    h = tf.matmul(phi,w)+bias
    size2 = tf.shape(h)

with tf.name_scope("Softmax") as scope:
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h,y_)
  cost = tf.reduce_sum(loss)
  loss_summ = tf.scalar_summary("cross entropy_loss", cost)

with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
#     The following block plots for every trainable variable
#      - Histogram of the entries of the Tensor
#      - Histogram of the gradient over the Tensor
#      - Histogram of the grradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
        
      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(h,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
   
merged = tf.merge_all_summaries()


# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

with tf.Session() as sess:
  with tf.device("/cpu:0"):
    print('Start session')
    writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/RBFN_tf/log_tb", sess.graph_def)

    step = 0
    sess.run(tf.initialize_all_variables())
    
#    #Debugging
#    batch_ind = np.random.choice(N,batch_size,replace=False)
#    result = sess.run([phi],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind]})
#    print(result[0])
    
    
    for i in range(max_iterations):
      batch_ind = np.random.choice(N,batch_size,replace=False)
      if i%100 == 1:
        #Measure train performance
        result = sess.run([cost,accuracy,train_step],feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind]})
        perf_collect[0,step] = result[0]
        perf_collect[2,step] = result[1]
        
        
        #Measure test performance
        test_ind = np.random.choice(Ntest,test_size,replace=False)
        result = sess.run([cost,accuracy,merged], feed_dict={ x: X_test[test_ind], y_: y_test[test_ind]})
        perf_collect[1,step] = result[0]
        perf_collect[3,step] = result[1]
      
        #Write information for Tensorboard
        summary_str = result[2]
        writer.add_summary(summary_str, i)
        writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
        
        #Print intermediate numbers to terminal
        acc = result[1]
        print("Estimated accuracy at iteration %s of %s: %s" % (i,max_iterations, acc))
        step += 1
      else:
        sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind]})
        
"""Additional plots"""
plt.figure()
plt.plot(perf_collect[2],label = 'Train accuracy')
plt.plot(perf_collect[3],label = 'Test accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(perf_collect[0],label = 'Train cost')
plt.plot(perf_collect[1],label = 'Test cost')
plt.legend()
plt.show()

# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ml_projects/RBFN_tf/log_tb