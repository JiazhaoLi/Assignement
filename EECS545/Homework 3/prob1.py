
# coding: utf-8

# In[1]:


import numpy as np
import math 
def count_label(labels):
    count_0 = 0
    count_1 = 0
    for i  in np.arange(np.shape(labels)[0]):
        if labels[i] == 0:
            count_0+=1
        else:
            count_1+=1
    return [count_0,count_1]

def count_cdm(feature,train_label):
    theta = np.zeros((4,np.shape(feature)[1]))
    for i in np.arange(np.shape(feature)[1]):
        count_1_dm0 = 0
        count_1_dm1 = 0
        count_0_dm0 = 0
        count_0_dm1 = 0
        for j in np.arange(np.shape(feature)[0]):
            if train_label[j] == 1:
                if feature[j][i] == 1:
                    count_1_dm1 += 1
                else:
                    count_1_dm0 += 1
            else:
                if feature[j][i] == 1:
                    count_0_dm1 += 1
                else:
                    count_0_dm0 += 1
            
        theta[0][i] = count_0_dm0
        theta[1][i] = count_0_dm1
        theta[2][i] = count_1_dm0
        theta[3][i] = count_1_dm1     
    return theta

def classifier(test_features,pi,N_cdm):
    pred_l = np.zeros((np.shape(test_labels)[0],1))
    print(np.shape(pred_l))
    for i in np.arange(np.shape(test_labels)[0]):
        p0 = pi[0]
        p1 = pi[1]
        for j in np.arange(np.shape(test_features)[1]):
            if test_features[i][j] == 1:
                p0 = p0*N_cdm[1][j]
                p1 = p1*N_cdm[3][j]
            else:
                p0 = p0*N_cdm[0][j]
                p1 = p1*N_cdm[2][j]
        if p1 > p0:
            pred_l[i,0] = 1
        else:
            pred_l[i,0] = 0
    return pred_l


# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")
# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")
N = np.shape(train_labels)[0]
d = np.shape(train_features)[1]
N_c = count_label(train_labels)
N_c = np.asarray(N_c)

N_cdm = count_cdm(train_features,train_labels)
N_cdm = np.asarray(N_cdm)

#print(N_cdm)
pi = (N_c)/(N)
N_cdm[0,:] = (N_cdm[0,:]+1)/(N_c[0]+2)
N_cdm[1,:]= (N_cdm[1,:]+1)/(N_c[0]+2)
N_cdm[2,:] = (N_cdm[2,:]+1)/(N_c[1]+2)
N_cdm[3,:] = (N_cdm[3,:]+1)/(N_c[1]+2)
#print(N_cdm)

pre = classifier(test_features,pi,N_cdm)
count_right = 0
for j in np.arange(np.shape(test_features)[0]):
    if pre[j]==test_labels[j]:
        count_right +=1
    else:
        continue
print('Error rate :',1-count_right/800)
#print(np.shape(train_features))
test_features = np.insert(test_features,0,1,axis = 1)
train_features = np.insert(train_features,0,1,axis = 1)
#print(np.shape(train_features))


# In[262]:




