import numpy as np
from sklearn.datasets import load_breast_cancer

# Normalized Binary dataset
# 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
X, y = load_breast_cancer().data, load_breast_cancer().target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
