# Name : Jiazhao Li Unique name: jiazhaol

import numpy as np
from sklearn import preprocessing
import sys
from sklearn import tree


def load_train_data(filename):
    SBD_traindata_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            word = line.split(' ')
            SBD_traindata_list.append([word[0], word[1], word[2]])
    return SBD_traindata_list


def load_test_data(filename):
    SBD_testdata_list = []
    with open(filename,'r') as f:
        for line in f:
            line = line.strip('\n')
            word = line.split(' ')
            SBD_testdata_list.append([word[0], word[1], word[2]])
    return SBD_testdata_list


def feature_label(data_list, mode):
    feature = []
    label = []
    index = 0
    for pair in data_list:
        if pair[2] == 'EOS' or pair[2] == 'NEOS':

            # label list
            if pair[2] == 'EOS':
                label.append(1)

            else:
                label.append(0)

            # label vacab
            L = data_list[index][1][:-1]
            if index == len(data_list)-1:
                R = ' '
            else:
                R = data_list[index + 1][1]

            len_L = int(len(L) < 3)


            if L =='':
                L_Cap = 0
            else:
                L_Cap = int(L[0].isupper())

            R_Cap = int(R[0].isupper())

            # own features
            LL_len = int(len(data_list[index-1][1]) > 3)
            if index == len(data_list)-2 or index == len(data_list)-1:
                RR_len = 0
            else:
                RR_len = int(len(data_list[index+1][1]) > 3)

            L_Cap_num  = 0
            for l in L :
                if l.isupper():
                    L_Cap_num += 1
            L_Cap_num = int(L_Cap_num > 3)


            if mode == 'CoreFeature':
                feature.append([L, R, len_L, L_Cap, R_Cap])
            elif mode == "OwnThree":
                feature.append([LL_len, RR_len, L_Cap_num])
            elif mode == 'CoreOwn':
                feature.append([L, R, len_L, L_Cap, R_Cap, LL_len, RR_len, L_Cap_num])
        index += 1
    return feature, label

# encode feature vector of
def encode_feature(train_feature,test_feature):
    word_dict = {}
    index = 2
    for pair in train_feature:
        if pair[0] not in word_dict:
            word_dict[pair[0]] = index
            index += 1
        if pair[1] not in word_dict:
            word_dict[pair[1]] = index
            index += 1

    for pair in test_feature:
        if pair[0] not in word_dict:
            word_dict[pair[0]] = index
            index += 1
        if pair[1] not in word_dict:
            word_dict[pair[1]] = index
            index += 1

    # substitute the feature vetor:
    for pair in train_feature:
        pair[0] = word_dict[pair[0]]
        pair[1] = word_dict[pair[1]]
    for pair in test_feature:
        pair[0] = word_dict[pair[0]]
        pair[1] = word_dict[pair[1]]

    Train_len = len(train_feature)

    all = train_feature + test_feature

    ohe = preprocessing.OneHotEncoder()  # Easier to read
    ohe.fit(all)
    Feature = ohe.transform(all).toarray()

    TrainEncode = Feature[:Train_len,:]
    TestEncode = Feature[Train_len:, :]
    return TrainEncode, TestEncode


def generate_outfile(SBDTestList, test_predict):
    with open('SBD.test.out', 'w') as f:
        test_predict_cate = []
        for label in test_predict:
            if label == 1:
                test_predict_cate.append('EOS')
            else:
                test_predict_cate.append('NEOS')

        f.write(mode + '\n')

        num = 0
        for pair in SBDTestList:
            if pair[2] == "EOS" or pair[2] == 'NEOS':
                f.write(" ".join([pair[0], pair[1], test_predict_cate[num]]))
                f.write('\n')
                num += 1
            else:
                f.write(" ".join([pair[0], pair[1], pair[2]]))
                f.write('\n')


if __name__ == '__main__':
    # train = "SBD.train"
    # test = "SBD.test"
    train = sys.argv[1]
    test = sys.argv[2]

    SBDTrainList = load_train_data(train)
    SBDTestList = load_test_data(test)


    ModeList = ['CoreFeature', "OwnThree", 'CoreOwn']

    # ModeList = ['CoreFeature']
    for mode in ModeList:
        train_feature, train_label  = feature_label(SBDTrainList, mode)
        test_feature, test_label = feature_label(SBDTestList, mode)

        TrainEncode, TestEncode = encode_feature(train_feature, test_feature)

        # train the Dicision Tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(TrainEncode, train_label)

        train_acc = clf.score(TrainEncode, train_label)
        test_acc = clf.score(TestEncode, test_label)
        test_predict = clf.predict(TestEncode)
        print(mode)
        print("train_acc: " + str(train_acc))
        print("test_acc: " + str(test_acc))

        if mode == 'CoreOwn':
            generate_outfile(SBDTestList, test_predict)








