
import codecs
import random
import re
import numpy as np
import csv 
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def load_train_data(train_data_path):
    file_object = codecs.open(train_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    X_c = []
    X_p = []
    Y = []
    for i, line in enumerate(lines[1:]):
        raw_list = line.strip().split('\t')
        X_c.append(raw_list[1]) 
        X_p.append(raw_list[2])
        Y.append(int(raw_list[0]))
    print('Finished load training data')
    return X_c,X_p,Y

def load_test_data(test_data_path):
    file_object = codecs.open(test_data_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    X_c = []
    X_p = []
    Y = []
    for i, line in enumerate(lines[1:]):
        raw_list = line.strip().split('\t')
        X_c.append(raw_list[1]) 
        X_p.append(raw_list[2])
        Y.append(raw_list[0])
    print('Finished load testing data')
    return X_c, X_p, Y
       

def TFIDFEmbedding(train_comments, test_comments, train_parent_comments, test_parent_comments):
    tfidf_char = TFIDF(min_df=1, 
           max_features=None,
           strip_accents='unicode',
           analyzer='char',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 8),  
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=0,
           stop_words='english')

   
    data_all = train_comments + test_comments
    tfidf_char.fit(data_all)
    train_X_char = tfidf_char.transform(train_comments)
    test_X_char = tfidf_char.transform(test_comments)


    tfidf_word = TFIDF(min_df=1, 
        max_features=None,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 8),  
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=0,
        stop_words='english')

    print('embedding word ......')

    tfidf_word.fit(data_all)
    train_X_word = tfidf_word.transform(train_comments)
    test_X_word = tfidf_word.transform(test_comments)
    
    print(np.shape(train_X_char))
    print(np.shape(train_X_word))
    print(np.shape(test_X_char))
    print(np.shape(test_X_word))
    train_X = sparse.hstack((train_X_char, train_X_word))
    test_X = sparse.hstack((test_X_char, test_X_word))
    

    print('TFIDF is completed !!!')
    print(np.shape(train_X))
    print(np.shape(test_X))
    return train_X, test_X


def LinearModel(X_train,Y_train,train_X, val_X, train_Y, val_Y, test_X):
    print('training linear regression model ......')
    LinearRegression = Ridge(alpha = 2.5)
    LinearRegression.fit(train_X, train_Y)
    print(train_Y.count(1))
    train_preds = np.array(LinearRegression.predict(train_X))
    
    train_preds = np.array([1 if x > 0.5 else 0 for x in train_preds]) 
    print(train_preds.tolist().count(1))


    val_preds = np.array(LinearRegression.predict(val_X))
    
    
    val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds]) 
    print(val_preds.tolist().count(1))
    print(val_preds)
    # print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(train_Y, train_preds, average='binary'))
    print('Recall score is:', recall_score(train_Y, train_preds, average='binary'))
    print('F1 score is:', f1_score(train_Y, train_preds, average='binary'))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    LinearRegression.fit(X_train,Y_train)
    preds = np.array(LinearRegression.predict(test_X))

    preds = np.array([1 if x > 0.5 else 0 for x in preds])


    return preds

def LogesticRegression(train_X, val_X, train_Y, val_Y, test_X):
    print('train logestic regression model')
    model_LoRe = LogisticRegressionCV(cv=5, random_state=0, class_weight='balanced')
    model_LoRe.fit(train_X, train_Y)
    train_preds = np.array(model_LoRe.predict(train_X))
    train_preds = np.array([1 if x > 0.5 else 0 for x in train_preds]) 
    val_preds = np.array(model_LoRe.predict(val_X))
    val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds]) 
    print('Precision score is:', precision_score(train_Y, train_preds, average='binary'))
    print('Recall score is:', recall_score(train_Y, train_preds, average='binary'))
    print('F1 score is:', f1_score(train_Y, train_preds, average='binary'))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    preds = np.array(model_LoRe.predict(test_X))
    preds = np.array([1 if x > 0.5 else 0 for x in preds])
    return preds

def DecisionTree(train_X, val_X, train_Y, val_Y, test_X):
    print('train decision tree')
    clf = tree.DecisionTreeClassifier(max_depth=4000)
    clf = clf.fit(train_X, train_Y)
    train_preds = np.array(clf.predict(train_X))
    # train_preds = np.array([1 if x > 0.5 else 0 for x in train_preds]) 
    val_preds = np.array(clf.predict(val_X))
    # val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds]) 
    print('Precision score is:', precision_score(train_Y, train_preds, average='binary'))
    print('Recall score is:', recall_score(train_Y, train_preds, average='binary'))
    print('F1 score is:', f1_score(train_Y, train_preds, average='binary'))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    clf = clf.fit(train_X+val_X, train_Y+val_Y)
    preds = np.array(clf.predict(test_X))
    # preds = np.array([1 if x > 0.5 else 0 for x in preds])
    return preds
    
def SVM_model(train_X, train_Y, X_train, y_train, X_test, y_test, test_X):
    
    clf = svm.SVC(C=1.0, kernel='rbf', probability=True, gamma='scale')
    clf.fit(X_train, y_train)
    val_preds = np.array(clf.predict(X_test))
    # val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds]) 
    print('Precision score is:', precision_score(y_test, val_preds, average='binary'))
    print('Recall score is:', recall_score(y_test, val_preds, average='binary'))
    print('F1 score is:', f1_score(y_test, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    clf.fit(train_X, train_Y)
    preds = np.array(clf.predict(test_X))
    # preds = np.array([1 if x > 0.5 else 0 for x in preds])
    return preds

def RFT_class(train_X, val_X, train_Y, val_Y, test_X):
    clf = RandomForestClassifier(n_estimators=100, max_depth=4000,random_state=0, class_weight='balanced', verbose=1)
    clf.fit(train_X, train_Y)
    train_preds = np.array(clf.predict(train_X))
    val_preds = np.array(clf.predict(val_X))
    print('Precision score is:', precision_score(train_Y, train_preds, average='binary'))
    print('Recall score is:', recall_score(train_Y, train_preds, average='binary'))
    print('F1 score is:', f1_score(train_Y, train_preds, average='binary'))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    
    clf = clf.fit(train_X+val_X, train_Y+val_Y)
    #print(model_LR.score(val_X, val_Y))
    preds = np.array(clf.predict(test_X))
    # preds = np.array([1 if x > 0.5 else 0 for x in preds])
    return preds


def SGD_Classifier(train_X, train_Y, X_train, y_train, X_test, y_test, test_X):
    from sklearn import linear_model
    clf = linear_model.SGDClassifier( alpha = 0.000025)
    clf.fit(X_train, y_train)
    train_preds = np.array(clf.predict(X_test))     
    print('Precision score is:', precision_score(y_test, train_preds, average='binary'))
    print('Recall score is:', recall_score(y_test, train_preds, average='binary'))
    print('F1 score is:', f1_score(y_test, train_preds, average='binary'))

    clf.fit(train_X, train_Y)
    preds = clf.predict(test_X)
    

    return preds

def fileWritting(output_path, preds):
    print('writting file ...')
    with open(output_path, 'w') as f:
        f = csv.writer(f, delimiter = ',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f.writerow(['Id', 'Category'])
        for i in (range(len(preds))):
            Id = str(i)
            Category = str(preds[i]) 
            f.writerow([Id, Category])
    print('writting is completed !!!')
            