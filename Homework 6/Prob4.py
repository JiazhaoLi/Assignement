
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
data, target = load_boston().data, load_boston().target
print("data_feature_size = "+str(np.shape(data)))
print("target size ="+str(np.shape(target)))
def normalize_feature(data):
    m,n = np.shape(data)
    data_processed = np.zeros((m,n))
    for i in range(n):
        mean = np.mean(data[:,i])
        var = np.std(data[:,i])
        data_processed[:,i] = (data[:,i]-mean)/var
    return data_processed

#do normalization:
data_processed = normalize_feature(data)
#print(np.shape(data_processed))

def PCA_me(data_processed):
    N,M= np.shape(data_processed)
    covariance = 1/N* data_processed.T@data_processed
    U, s, V = np.linalg.svd(covariance, full_matrices=True)
    return U

pca_me = PCA_me(data_processed)
projection = data_processed@pca_me[:,0:2]
print("projected data = "+ str(np.shape(projection)))
print("first two principal direction is:"+"\n"+str(pca_me[:,0:2]))
plt.scatter(projection[:,0],projection[:,1],c = target / max(target))


pca = PCA(n_components=2)
pca.fit(data_processed)
print(pca.explained_variance_ratio_)
newA=pca.transform(data_processed) 
print(newA[:,0:2])
print(np.shape(newA))
plt.scatter(newA[:,0],newA[:,1],c = target / max(target))

