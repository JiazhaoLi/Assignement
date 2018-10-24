
# coding: utf-8

# In[3]:


import numpy as np

from sklearn import datasets, svm
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='./')

images = mnist.data
targets = mnist.target

N = len(images)
np.random.seed(1234)
inds = np.random.permutation(N)
images = np.array([images[i] for i in inds])
targets = np.array([targets[i] for i in inds])

# Normalize data
X_data = images/255.0
Y = targets

# Train/test split
X_train, y_train = X_data[:10000], Y[:10000]
X_test, y_test = X_data[-10000:], Y[-10000:]
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

classifier = svm.SVC(C=1,gamma=1,kernel='rbf')
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)

temp = np.arange(X_train.shape[0])
np.random.shuffle(temp)
x = X_train[temp]
y = y_train[temp]

n = int(len(X_train)/5)

x_1, y_1 = x[:n], y[:n]
x_2, y_2 = x[n:n*2], y[n:n*2]
x_3, y_3 = x[n*2:n*3], y[n*2:n*3]
x_4, y_4 = x[n*3:n*4], y[n*3:n*4]
x_5, y_5 = x[n*4:], y[n*4:]

train_1_x = np.vstack((x_1,x_2,x_3,x_4))
train_1_y = np.hstack((y_1,y_2,y_3,y_4))

train_2_x = np.vstack((x_1,x_2,x_3,x_5))
train_2_y = np.hstack((y_1,y_2,y_3,y_5))

train_3_x = np.vstack((x_1,x_2,x_4,x_5))
train_3_y = np.hstack((y_1,y_2,y_4,y_5))

train_4_x = np.vstack((x_1,x_3,x_4,x_5))
train_4_y = np.hstack((y_1,y_3,y_4,y_5))

train_5_x = np.vstack((x_2,x_3,x_4,x_5))
train_5_y = np.hstack((y_2,y_3,y_4,y_5))

accuracy = np.zeros((3,4))

C = [1,3,5]
gamma = [0.05,0.1,0.5,1]
for i in range(3):
    for j in range(4):
        classifier = svm.SVC(C=C[i],gamma=gamma[j],kernel='rbf')
        classifier.fit(train_1_x, train_1_y)
        ac1 = classifier.score(x_5,y_5)
        
        classifier = svm.SVC(C=C[i],gamma=gamma[j],kernel='rbf')
        classifier.fit(train_2_x, train_2_y)
        ac2 = classifier.score(x_4,y_4)        
        
        classifier = svm.SVC(C=C[i],gamma=gamma[j],kernel='rbf')
        classifier.fit(train_3_x, train_3_y)
        ac3 = classifier.score(x_3,y_3)
        
        classifier = svm.SVC(C=C[i],gamma=gamma[j],kernel='rbf')
        classifier.fit(train_4_x, train_4_y)
        ac4 = classifier.score(x_2,y_2)
        
        classifier = svm.SVC(C=C[i],gamma=gamma[j],kernel='rbf')
        classifier.fit(train_5_x, train_5_y)
        ac5 = classifier.score(x_1,y_1)
        
        accuracy[i,j] = (ac1 + ac2 + ac3 + ac4 + ac5)/5
print(accuracy)

classifier = svm.SVC(C=3,gamma=0.05,kernel='rbf')
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)

