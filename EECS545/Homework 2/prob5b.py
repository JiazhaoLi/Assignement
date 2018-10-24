import numpy as np

data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)
