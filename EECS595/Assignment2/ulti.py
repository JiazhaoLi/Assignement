import pdb
# Jiazhao Li (unique name: jiazhaol)

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from numpy import linalg as LA

import random

from tqdm import tqdm

def load_train_data(filepath='./languageIdentification.data/train'):
    # filepath = './languageIdentification.data/train'
    index = 0
    seq_index = 0
    with open(filepath, encoding='utf-8') as f:
        train_sentence_len = []
        train_label = []
        train_feature =[]
        for line in f:
            index +=1
            line = line.strip('\n')
            words = line.split(' ')[:-1]
            if words[0].lower() == 'english':
                label = [1, 0, 0]
            elif words[0].lower() == 'french':
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            # string of sentence
            content_string = []
            for word in words[1:]:
                content_string.extend(word.lower())
                content_string.extend(' ')
            
            content_string = content_string[:-1]
            # if len(content_string)<5:
            #     print(content_string)
            # form 5 characters maintain the white space
            for i in range(len(content_string)-4):
                if i+5 <= len(content_string):
                    char_5 = content_string[i: i+5]
                    train_feature.append(char_5)
                    train_label.append(label)
            train_sentence_len.append(i+1)

    return train_feature, train_label, train_sentence_len


def load_val_data(filepath='./languageIdentification.data/dev'):
    # filepath = './languageIdentification.data/dev'
    with open(filepath, encoding='utf-8') as f:
        val_sentence_len = []
        val_label = []
        val_feature =[]
        for line in f:
            line = line.strip('\n')
            words = line.split(' ')[:-1]
            if words[0].lower() == 'english':
                label = [1, 0, 0]
            elif words[0].lower() == 'french':
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            # string of sentence
            content_string = []
            for word in words[1:]:
                content_string.extend(word.lower())
                content_string.extend(' ')
            content_string = content_string[:-1]
            # form 5 characters maintain the white space
            for i in range(len(content_string)-4):
                if i+5 <= len(content_string):
                    char_5 = content_string[i: i+5]
                    val_feature.append(char_5)
                    val_label.append(label)
            val_sentence_len.append(i+1)

    return val_feature, val_label, val_sentence_len


def load_test_data(filepath='./languageIdentification.data/test'):
    # filepath = './languageIdentification.data/test'
    with open(filepath, encoding="ISO-8859-1") as f:
        test_feature = []
        len_sentence =[]
        for line in f:
            words = line.strip().split()
            content_string = []
            for word in words[1:]:
                content_string.extend(word.lower())
                content_string.extend(' ')
            content_string = content_string[:-1]
            len_char5 = 0
            # form 5 characters maintain the white space
            for i in range(len(content_string) - 4):
                if i + 5 <= len(content_string):
                    char_5 = content_string[i: i + 5]
                    test_feature.append(char_5)
                    len_char5+=1
            len_sentence.append(len_char5)

    filepath = './languageIdentification.data/test_solutions'
    with open(filepath, encoding="us-ascii") as f:
        test_label = []
        for line in f:
            words = line.strip().split()
            if words[1] == 'Italian':
                label =2
            elif words[1] == 'English':
                label =0
            else:
                label = 1
            test_label.append(label)
    test_label = np.array(test_label)

    return test_feature, test_label, len_sentence


def one_hot_encode(train_feature, train_label, test_feature, val_feature):
    char_dict = dict()
    index = 1
    for c5 in train_feature:
        for c in c5:
            if c not in char_dict:
                char_dict[c] = index
                index += 1
    # print(char_dict)
    # transform the train_feature and trainlabel
    train_feature_transfer = []
    for c5 in train_feature:
        c5_new = []
        for c in c5:
            c5_new.append(char_dict[c])
        train_feature_transfer.append(c5_new)
    # transform the test_feature
    test_feature_transfer = []
    for c5 in test_feature:
        c5_new = []
        for c in c5:
            if c not in char_dict:
                c5_new.append(0)
            else:
                c5_new.append(char_dict[c])
        test_feature_transfer.append(c5_new)
    # transform the val feature
    val_feature_transfer = []
    for c5 in val_feature:
        c5_new = []
        for c in c5:
            if c not in char_dict:
                c5_new.append(0)
            else:
                c5_new.append(char_dict[c])
        val_feature_transfer.append(c5_new)

    f1 = OneHotEncoder(handle_unknown='ignore')
    f1.fit(train_feature_transfer)
    train_feature_encode = f1.transform(train_feature_transfer)
    test_feature_encode = f1.transform(test_feature_transfer)
    val_feature_encode = f1.transform(val_feature_transfer)
    return train_feature_encode.toarray(), np.array(train_label), test_feature_encode.toarray(), val_feature_encode.toarray()


def parameter_init(D, input_dim):
    # size of parameters
    # W1 : D * 5c
    # b1 : D * 1
    # W2 : 3 * D
    # b2 : 3 * 1
    np.random.seed(7)

    parameter = dict()
    parameter['W1'] = np.random.normal(0, 1, (D, input_dim))
    parameter['b1'] = np.ones((D, 1))

    parameter['W2'] = np.random.normal(0, 1, (3, D))
    parameter['b2'] = np.ones((3, 1))

    return parameter


def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))


def Softmax(predict_label):
    sum_all = np.sum(np.exp(predict_label))
    return [np.exp(x)/sum_all for x in predict_label]


def forward_proporgation(parameters, X):
    # input is 5c * N
    # W1 : D * 5c
    # b1 : D * 1
    # W2 : 3 * D
    # b2 : 3 * 1
    # output 3 * N
    y1 = np.matmul(parameters['W1'], X).reshape(-1,1) + parameters['b1'].reshape(-1,1)
    A1 = sigmoid(y1)

    y2 = np.matmul(parameters['W2'], A1).reshape(-1,1) + parameters['b2'].reshape(-1,1)
    predict_label = Softmax(y2)

    cache = [y1, A1, y2, parameters]
    return cache, np.array(predict_label)


def loss_function(predict_label, label):

    return 1/2 * LA.norm(predict_label-label.reshape(-1,1), 2) ** 2


def back_proporgation(X, cache, predict_label, label):

    [y1, A1, y2, parameters] = cache

    W2 = parameters['W2']

    dp = predict_label - label.reshape(-1,1)
    dy2 = np.matmul(np.diag(predict_label[:,0])-np.matmul(predict_label, predict_label.T), dp)

    dW2 = np.matmul(dy2, A1.T)
    db2 = dy2

    dA1 = np.matmul(W2.T, dy2)
    dy1 = np.multiply(dA1, np.multiply(A1, (1 - A1)))
    dW1 = np.matmul(dy1, X.T)
    db1 = dy1

    gradient = {'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}

    return gradient


def update_parameter(parameter, gradient, lr):
    parameter['W1'] = parameter['W1'] - lr * gradient['dW1']
    parameter['b1'] = parameter['b1'] - lr * gradient['db1']
    parameter['W2'] = parameter['W2'] - lr * gradient['dW2']
    parameter['b2'] = parameter['b2'] - lr * gradient['db2']
    
    return parameter


def main_train(train_feature_encode, train_label_encode, parameter, lr=0.1):

    train_index = np.arange(0, np.shape(train_feature_encode)[0])

    random.shuffle(train_index)
    for i in tqdm(train_index):
        cache, predict_label = forward_proporgation(parameter, train_feature_encode[i, :].T)
        cost = loss_function(predict_label, train_label_encode[i, :])

        gradient = back_proporgation(train_feature_encode[i, :].reshape(-1, 1), cache, predict_label,
                                     train_label_encode[i, :])
        
        parameter = update_parameter(parameter, gradient, lr)
        
    return parameter


def train_prediction(train_feature_encode, train_label, train_sentence_len,parameter):

    predict_label_list = []
    true_label_list = []
    for i in tqdm(range(np.shape(train_feature_encode)[0])):
        cache, predict_label = forward_proporgation(parameter, train_feature_encode[i, :].T)
        train_predict_label = predict_label.tolist()
        train_predict_label = train_predict_label.index(max(train_predict_label))
        train_true_label = train_label[i].index(max(train_label[i]))
        predict_label_list.append(train_predict_label)
        true_label_list.append(train_true_label)


    end = 0
    count = 0
    for index in tqdm(range(len(train_sentence_len))):
        start = end
        end += train_sentence_len[index]

        sentence_predict = predict_label_list[start:end]
        sentence_true = true_label_list[start:end]

        if len(sentence_predict) < 5:
            continue
        else:
            majority_predict_label = np.argmax(np.bincount(sentence_predict))
            majority_true_label = np.argmax(np.bincount(sentence_true))

        if majority_predict_label == majority_true_label:
            count += 1
   

    return count/(index + 1)


def val_prediction(val_feature_encode, val_label, val_sentence_len, parameter):

    predict_label_list = []
    true_label_list = []
    for i in tqdm(range(np.shape(val_feature_encode)[0])):
        cache, predict_label = forward_proporgation(parameter, val_feature_encode[i, :].T)
        train_predict_label = predict_label.tolist()
        train_predict_label = train_predict_label.index(max(train_predict_label))
        train_true_label = val_label[i].index(max(val_label[i]))
        predict_label_list.append(train_predict_label)
        true_label_list.append(train_true_label)
    end = 0
    count = 0
    for index in tqdm(range(len(val_sentence_len))):
        start = end
        end += val_sentence_len[index]

        sentence_predict = predict_label_list[start:end]
        sentence_true = true_label_list[start:end]

        if len(sentence_predict) < 5:
            continue
        else:
            majority_predict_label = np.argmax(np.bincount(sentence_predict))
            majority_true_label = np.argmax(np.bincount(sentence_true))

        if majority_predict_label == majority_true_label:
            count += 1
    return count / (index + 1)


def test_prediction(test_feature_encode, test_label, test_sentence_len,parameter):
    content = []
    with open('./languageIdentification.data/test', encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            content.append(line)
    with open('./languageIdentificationPart1.output','w') as f:
        print('Generate output profile')
        predict_label_list = []
        test_lable_list = []

        for i in tqdm(range(np.shape(test_feature_encode)[0])):
            cache, predict_label = forward_proporgation(parameter, test_feature_encode[i, :].T)
            test_predict_label = predict_label.tolist()
            test_predict_label = test_predict_label.index(max(test_predict_label))
            predict_label_list.append(test_predict_label)
        end = 0
        count = 0
        for index in tqdm(range(len(test_sentence_len))):
            start = end
            end += test_sentence_len[index]

            sentence_predict = predict_label_list[start:end]

            if len(sentence_predict) < 5:
                f.write(content[index]+' '+'Not long Enough'+'\n')
                continue
            else:
                majority_predict_label = np.argmax(np.bincount(sentence_predict))
                if majority_predict_label == 0:
                    f.write(content[index] + ' ' + "English"+'\n')
                elif majority_predict_label == 1:
                    f.write(content[index] + ' ' + "French"+'\n')
                else:
                    f.write(content[index] + ' ' + "Italian"+'\n')
            if majority_predict_label == test_label[index]:
                count += 1

    return count / (index + 1)
