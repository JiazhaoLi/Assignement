# Jiazhao Li (unique name: jiazhaol)
from ulti import *
import random
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def main():
    epoch = 3
    train_feature, train_label, train_sentence_len = load_train_data(sys.argv[1])  # 7781 sentences
    val_feature, val_label, val_sentence_len = load_val_data(sys.argv[2])
    test_feature, test_label, test_sentence_len = load_test_data(sys.argv[3])
    print('Finished data loading')


    train_feature_encode, train_label_encode, test_feature_encode, val_feature_encode = one_hot_encode(train_feature, train_label, test_feature, val_feature)
    print('Finished data encoding')

    D = 100
    d = 81
    lr = 0.1
    train_acc_list = []
    val_acc_list = []

    print("Initialized the parameters")
    parameter = parameter_init(D, 5*d)



    # Before _training
    print("Pre-result")
    train_acc = train_prediction(train_feature_encode, train_label, train_sentence_len, parameter)
    val_acc = val_prediction(val_feature_encode, val_label, val_sentence_len, parameter)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    print('train_acc = ' + str(train_acc))
    print("val_acc=" + str(val_acc))

    # start Training
    for i in range(epoch):
        print('Start {} training process'.format(str(i+1)))
        parameters = main_train(train_feature_encode, train_label_encode, parameter, lr)
        
    # start Prediction
        train_acc = train_prediction(train_feature_encode, train_label, train_sentence_len, parameters)
        print('train_acc = ' + str(train_acc))

        val_acc = val_prediction(val_feature_encode, val_label, val_sentence_len, parameters)
        print("val_acc=" + str(val_acc))

        test_acc = test_prediction(test_feature_encode, test_label, test_sentence_len, parameters)
        print("test_acc = " + str(test_acc))

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    
    print(train_acc_list)
    print(val_acc_list)
    plt.plot([0, 1, 2, 3], train_acc_list)
    plt.plot([0, 1, 2, 3], val_acc_list)
    plt.legend(['train', 'dev'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('Train and Validation accuracy for each epoches')
    plt.savefig('accuracy.png')

    # part2 tuning part
    # lr and d
    tuning_devlist = []
    tuning_testlist = []
    parameter_list = [[0.1,50],[0.1,150],[0.1,200],[0.05,100],[0.01,100]]
    for para_pair in parameter_list:
        parameter = parameter_init(para_pair[1], 5*d)
        # start Training
        for i in range(epoch):
            print('Start setting{}, {}epoch training process'.format(str(para_pair),str(i+1)))
            parameters = main_train(train_feature_encode, train_label_encode, parameter, lr)

        # start Prediction
        val_acc = val_prediction(val_feature_encode, val_label, val_sentence_len, parameters)
        print("val_acc=" + str(val_acc))
        tuning_devlist.append(val_acc)

        test_acc = test_prediction(test_feature_encode, test_label, test_sentence_len, parameters)
        print("test_acc = " + str(test_acc))
        tuning_testlist.append(test_acc)
    print("tuning result on val set: " +str(tuning_devlist))
    print("tuning result on test set: " +str(tuning_testlist))
    best_set = tuning_testlist[np.argmax(tuning_devlist)]
    print(best_set)

            



if __name__ == '__main__':
    main()






