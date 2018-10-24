
# from sklearn.preprocessing import OneHotEncoder
# # enc = OneHotEncoder()
# # a = [['a', 0, 3], ['b', 1, 0], ['v', 2, 1], ['c', 0, 2]]
# # enc.fit(a)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
a = [['a', 0, 3], ['b', 1, 0], ['c', 2, 1], ['d', 0, 2]]
enc.fit(a)


# enc.n_values_
#
# enc.feature_indices_
#
v = enc.transform(a).toarray()
print(v[:2,:])