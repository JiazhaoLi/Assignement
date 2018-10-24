import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
# data and target are given below 
# data is a numpy array consisting of 100 2-dimensional points
# target is a numpy array consisting of 100 values of 1 or -1
data = np.ones((100, 2))
data[:,0] = np.random.uniform(-1.5, 1.5, 100)
data[:,1] = np.random.uniform(-2, 2, 100)
z = data[:,0] ** 2 + ( data[:,1] - (data[:,0] ** 2) ** 0.333 ) ** 2  
target = np.asarray( z > 1.5, dtype = int) * 2 - 1




def f(x,y,data,target):
    print(x.size)
    print(data[0][0].size)
    for i in range(len(data)):
        tmp = 0
        for j in range(len(data)):
            norm = (data[j][0]-x)*(data[j][0]-x)+(data[j][1]-y)*(data[j][1]-y)
            tmp += a[j]*target[j]*np.exp(-norm/(2*sigma*sigma))
        return tmp

a = np.zeros(100)
sigma = 0.1

for _ in range(10):
    for i in range(len(data)):
        tmp = 0
        for j in range(len(data)):
            norm = (data[j][0]-data[i][0])*(data[j][0]-data[i][0])+(data[j][1]-data[i][1])*(data[j][1]-data[i][1])
            tmp += a[j]*target[j]*np.exp(-norm/(2*sigma*sigma))
        if tmp*target[i]>0:
            pass
        else:
            a[i] += 1

#print(a)

for i in range(len(data)):
    if target[i]==1:
        plt.scatter(data[i,0], data[i,1], color="red")
    if target[i]==-1:
        plt.scatter(data[i, 0], data[i, 1], color="blue")

h = 0.05
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.contour(xx, yy, f(xx,yy,data,target),levels=[0])

plt.title('decision boundary learned by Gaussian Kernel Perceptron with σ = 0.1')
plt.show()

sigma = 1.0

for _ in range(30):
    for i in range(len(data)):
        tmp = 0
        for j in range(len(data)):
            norm = (data[j][0]-data[i][0])*(data[j][0]-data[i][0])+(data[j][1]-data[i][1])*(data[j][1]-data[i][1])
            tmp += a[j]*target[j]*np.exp(-norm/(2*sigma*sigma))
        if tmp*target[i]>0:
            pass
        else:
            a[i] += 1


for i in range(len(data)):
    if target[i]==1:
        plt.scatter(data[i,0], data[i,1], color="red")
    if target[i]==-1:
        plt.scatter(data[i, 0], data[i, 1], color="blue")

h = 0.05
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.contour(xx, yy, f(xx,yy,data,target),levels=[0])

plt.title('decision boundary learned by Gaussian Kernel Perceptron with σ = 1.0')
plt.show()




#print(a)