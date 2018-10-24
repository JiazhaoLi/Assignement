import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

for k in range(100):

  # Shuffle data
  rand_perm = np.random.permutation(Ndata)
  features = [features_orig[ind] for ind in rand_perm]
  labels = [labels_orig[ind] for ind in rand_perm]

  # Train/test split
  Nsplit = 50
  X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
  X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

  # Preprocess your data - Normalization, adding a constant feature
  params = preproc_params(X_train)
  X_train = preprocess(X_train, params)
  X_test = preprocess(X_test, params)

  # Solve for optimal w
  # Use your solver function
  w = solve(X_train, y_train)

  # Collect train and test errors
  # Use your implementation of the mse function
  train_errs.append(mse(X_train, y_train, w))
  test_errs.append(mse(X_test, y_test, w))

print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))
