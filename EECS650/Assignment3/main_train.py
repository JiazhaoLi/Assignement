from util import *
import numpy as np
train_data_path = './data/train.tsv'
test_data_path = './data/test.tsv'
output_path = './data/submission.csv'

train_X_c, train_X_p, train_Y = load_train_data(train_data_path)
test_X_c, test_X_p, Index = load_test_data(test_data_path)

print('Data loaded')

# train_X_comments, test_X_comments = wordCountFeature(train_X_c,test_X_c,train_X_p,test_X_p)

train_X, test_X = TFIDFEmbedding(train_X_c, test_X_c, train_X_p,test_X_p)

# num_data = np.shape(train_X)[0]
# print(num_data)
# exit()
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.1, random_state=42)


print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))
# SVM(train_X, test_X, train_Y)


# predict_test = SGD_Classifier(train_X, train_Y, X_train, y_train, X_test, y_test, test_X)

predict_test = LinearModel(train_X,train_Y,X_train, X_test, y_train, y_test, test_X)
# fileWritting(output_path, predict_test)
# predict_test = LogesticRegression(X_train, X_test, y_train, y_test, test_X)
# predict_test = RFT_class(X_train, X_test, y_train, y_test, test_X)


# predict_test = DecisionTree(X_train, X_test, y_train, y_test, test_X)

# predict_test= SVM_model(train_X, train_Y, X_train, y_train, X_test, y_test, test_X)
# predict_test = XgbModel(X_train, X_test, y_train, y_test, test_X)
fileWritting(output_path, predict_test)
# print(preds_LR.tolist().count(0), preds_LR.tolist().count(1)) 