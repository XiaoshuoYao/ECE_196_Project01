import mxnet as mx
from mxnet import nd
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
import random
from IPython import display
from matplotlib import pyplot as plt

 
columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]#represents the columns of selected features and labels 
datas = pd.read_csv( 'C:/Users/twovo/Desktop/GrandMaster_Ranked_Games.csv' )#read data as dataframe object
#seperate the data set for training and testing
data_array = datas.iloc[:8000, columns].values 
test_data_array = datas.iloc[8001:, columns].values
#create ndarray for features&labels
features = nd.array(data_array[:,1:], ctx=mx.gpu(0))
labels = nd.array(data_array[:,0], ctx=mx.gpu(0))
test_features = nd.array(test_data_array[:,1:], ctx=mx.gpu(0))
test_labels = nd.array(test_data_array[:,0], ctx=mx.gpu(0))
#define batch size, learning rate and epoch times
batch_size = 100
lr = 0.00002
num_epochs = 4000

#data iterater 
#randomly select samples from training set and
#construct a small bacth of trainning data 
def data_iter(batch_size, features, labels):
    input_line = len(features)
    indices = list(range(input_line))
    random.shuffle(indices)  
    for i in range(0, input_line, batch_size):
        j = nd.array(indices[i: min(i + batch_size, input_line)],ctx=mx.gpu(0))
        yield features.take(j), labels.take(j)

#create the weight and bias as a nd array
w = nd.random.uniform(shape=(15, 1), ctx=mx.gpu(0))
b = nd.random.uniform(shape=(1,), ctx=mx.gpu(0))
#attach gradient for future calculation
w.attach_grad()
b.attach_grad()

#regression model
#define a linear regression model
def linear_regression_model(X, w, b):   
    return nd.dot(X, w) + b
#loss function
#define a square loss function
def squared_loss(y_predict, y):   
    return (y_predict - y.reshape(y_predict.shape)) ** 2 / 2
#learning function
#define a stochastic gradient descent
def sgd(params, lr, batch_size):   
    for param in params:
        param[:] = param - lr * param.grad / batch_size

 
net = linear_regression_model
loss = squared_loss
 
# generate random batch of samples
# compare the predicted y and real y
# imply sgd method to learn 
for epoch in range(num_epochs):   
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)   
        l.backward()   
        sgd([w, b], lr, batch_size)   
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, sqr_loss %f' % (epoch + 1, train_l.mean().asnumpy()))
#print the weights and the bias
print(w)
print(b)

#define the total lost
lost = 0
#use the test samples to calculate the accuracy 
for i in range(test_labels.size):
   lost = lost+(nd.dot(test_features[i],w) + b-(test_labels[i]))/test_labels[i]/test_labels.size
#print accuracy
print('avrage lost: ',lost)
 
 
##########################################Results###################################################

#[[ 13.172637 ]
# [ 18.63996  ]
# [  7.82816  ]
# [-10.893283 ]
# [ -5.654686 ]
# [-29.799541 ]
# [ 21.447948 ]
# [ 20.387068 ]
# [ 19.900015 ]
# [ -5.617373 ]
# [  6.8405666]
# [  6.5887628]
# [ -1.6300768]
# [ 13.137215 ]
# [  1.7602258]]
#<NDArray 15x1 @gpu(0)>

#[409.96112]
#<NDArray 1 @gpu(0)>
#avrage lost:
#[-0.00651435]
#<NDArray 1 @gpu(0)>

#[[ 34.459106  ]
# [ 24.83606   ]
# [ 12.759547  ]
# [ -7.2867966 ]
# [  0.20280527]
# [-24.666992  ]
# [ 19.32646   ]
# [ 12.705511  ]
# [ 19.707108  ]
# [ -9.082143  ]
# [  7.0535207 ]
# [  6.526523  ]
# [ -1.761117  ]
# [ 14.1816635 ]
# [  1.5776412 ]]
#<NDArray 15x1 @gpu(0)>

#[367.81024]
#<NDArray 1 @gpu(0)>
#avrage lost:
#[-0.0116778]
 



