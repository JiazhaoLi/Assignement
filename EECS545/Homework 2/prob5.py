import numpy as np

data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1
