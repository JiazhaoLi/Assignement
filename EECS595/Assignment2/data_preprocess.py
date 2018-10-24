# Jiazhao Li (unique name: jiazhaol)

from ulti import *
import random
import pickle
import sys
import matplotlib.pyplot as plt

def main():
    epoch = 3
    train_feature, train_label, train_sentence_len = load_train_data(sys.argv[1])  # 7781 sentences
    val_feature, val_label, val_sentence_len = load_val_data(sys.argv[2])
    test_feature, test_label, test_sentence_len = load_test_data(sys.argv[3])
    for i in train_feature:
        if len(i)<5:
            print(1)

    print('Finished data loading')
    train_feature_encode, train_label_encode, test_feature_encode, val_feature_encode = one_hot_encode(train_feature, train_label, test_feature, val_feature)
    print('Finished data encoding')
    D = 100
    d = 81
    train_acc_list = []
    val_acc_list = []

    print("Initialized the parameters")
    parameter = parameter_init(D, 5*d)

    # Before _training
    print("pre-result")
    train_acc = train_prediction(train_feature_encode, train_label, train_sentence_len, parameter)
    val_acc = val_prediction(val_feature_encode, val_label, val_sentence_len, parameter)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    print('train_acc = ' + str(train_acc))
    print("val_acc=" + str(val_acc))

    # start Training
    print('Start training process')
    for i in range(epoch):
        parameters = main_train(train_feature_encode, train_label_encode, parameter, lr=0.1)
        print('Saveing model' + str(i+1))
        output = open('model' + str(i+1) + '.pkl', 'wb')
        pickle.dump(parameters, output)
        output.close()

    # start Prediction
    for i in range(epoch):
        print("Load model" + str(i + 1))
        with open('model' + str(i + 1) + '.pkl', 'rb') as pkl_file:
            parameters = pickle.load(pkl_file)

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

    # part2 validation part

    D_list = [70,120,140,160,180]
    lr_list = [0.01, 0.03, 0.05, 0.07, 0.15]

    for set in range(5):
        D = D_list[set]
        d = 81
        lr = lr_list[set]
        train_acc_list = []
        val_acc_list = []

        print("Initialized the parameters")
        parameter = parameter_init(D, 5 * d)

        # Before _training
        print("pre-result")
        train_acc = train_prediction(train_feature_encode, train_label, train_sentence_len, parameter)
        val_acc = val_prediction(val_feature_encode, val_label, val_sentence_len, parameter)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print('train_acc = ' + str(train_acc))
        print("val_acc=" + str(val_acc))

        # start Training
        print('Start training process')
        for i in range(epoch):
            parameters = main_train(train_feature_encode, train_label_encode, parameter, lr)
            print('Saveing model' + str(i + 1))
            with open('model' + str(i + 1)+str(D)+ str(lr) + '.pkl', 'wb') as output:
                pickle.dump(parameters, output)

        # start Prediction
        for i in range(epoch):
            print("Load model" + str(i + 1))
            with open('model' + str(i + 1)+str(D)+ str(lr) + '.pkl', 'rb') as pkl_file:
                parameters = pickle.load(pkl_file)

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
        plt.title('Epoch vs accuracy')
        plt.savefig('accuracy'+str(D)+ str(lr)+'.png')




if __name__ == '__main__':
    main()






