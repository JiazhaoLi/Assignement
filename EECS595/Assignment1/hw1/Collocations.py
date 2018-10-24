from __future__ import division
# Name : Jiazhao Li Unique name: jiazhaol
import string
import numpy as np
import sys


def generate_unigrams_bigrams():
    unigrams = {}
    bigrams = {}
    unigrams_num = 0
    bigrams_num = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(' ')
            # delete ''
            for token in tokens:
                if token =="":
                    tokens.remove(token)
            # delete punctuation and form unigrams
            for token in tokens:
                if token not in string.punctuation:
                    unigrams_num += 1
                    if token not in unigrams:
                        unigrams[token] = 1
                    else:
                        unigrams[token] += 1

            # for bigrams)list
            bigrams_list = []
            for index in range(len(tokens)-1):
                bigrams_list.append(' '.join(tokens[index:index+2]))
            # form bigrams
            for pair in bigrams_list:
                word = pair.split(' ')
                if word[0] not in string.punctuation and word[1] not in string.punctuation:
                    bigrams_num += 1
                    if pair not in bigrams:
                        bigrams[pair] = 1
                    else:
                        bigrams[pair] += 1

    return unigrams, unigrams_num, bigrams, bigrams_num


def hashmap(bigrams):
    L_RowDict = {}
    R_ColDict = {}
    RowIndex = 0
    ColIndex = 0
    for pair, val in bigrams.items():
        word = pair.split(' ')
        if word[0] not in L_RowDict:
            L_RowDict[word[0]] = RowIndex
            RowIndex += 1
        if word[1] not in R_ColDict:
            R_ColDict[word[1]] = ColIndex
            ColIndex += 1
    cross = np.zeros((RowIndex,ColIndex))

    for pair, val in up5_bigrams.items():
        word = pair.split(' ')
        cross[L_RowDict[word[0]],R_ColDict[word[1]]] = val

    return cross,L_RowDict,R_ColDict


def Chi_sqaure(cross,L_RowDict,R_ColDict,up5_bigrams, bigrams_num):
    Observation = np.zeros((2,2))
    Expectation = np.zeros((2,2))
    pair_chi = {}
    for pair, val in up5_bigrams.items():
        chi2 = 0
        word = pair.split(' ')
        Observation[0, 0] = cross[L_RowDict[word[0]], R_ColDict[word[1]]]
        Observation[0, 1] = np.sum(cross[:, R_ColDict[word[1]]]) - Observation[0, 0]
        Observation[1, 0] = np.sum(cross[L_RowDict[word[0]], :]) - Observation[0, 0]
        Observation[1, 1] = np.sum(cross) - Observation[0, 0]-Observation[0, 1] - Observation[1, 0]

        Expectation[0, 0] = (Observation[0, 0] + Observation[0, 1]) * (Observation[0, 0] + Observation[1, 0]) / bigrams_num
        Expectation[1, 0] = (Observation[1, 0] + Observation[0, 0]) * (Observation[1, 0] + Observation[1, 1]) / bigrams_num
        Expectation[0, 1] = (Observation[0, 1] + Observation[0, 0]) * (Observation[0, 1] + Observation[1, 1]) / bigrams_num
        Expectation[1, 1] = (Observation[1, 1] + Observation[0, 1]) * (Observation[1, 1] + Observation[1, 0]) / bigrams_num

        for i in range(2):
            for j in range(2):
                chi2 += (Observation[i, j] - Expectation[i, j])**2 / Expectation[i,j]
        pair_chi[pair] = chi2
    return pair_chi


def PMI(unigrams,up5_bigrams,unigrams_num,bigrams_num):
    PMI_pair = {}
    for pair,val in up5_bigrams.items():
        P_pair = val/bigrams_num
        word = pair.split(' ')
        L = word[0]
        R = word[1]
        P_each = (float(unigrams[L])/ unigrams_num ) * (float(unigrams[R])/ unigrams_num)
        PMI_pair[pair] = np.log(P_pair/float(P_each))
    return PMI_pair


if __name__ == '__main__':

    filename = sys.argv[1]
    unigrams, unigrams_num, bigrams, bigrams_num = generate_unigrams_bigrams()

    # filter the less than 5
    up5_bigrams = {}
    up5_bigrams_num = 0
    for pair, num in bigrams.items():
        if num >= 5:
            up5_bigrams[pair] = num
            up5_bigrams_num += num

    cross, L_RowDict, R_ColDict = hashmap(up5_bigrams)


    pair_chi = Chi_sqaure(cross, L_RowDict, R_ColDict, up5_bigrams, up5_bigrams_num)

    top20_Chi = sorted(pair_chi.items(), key=lambda kv: kv[1], reverse=True)[:20]

    for pair in top20_Chi:
        print(pair)

    # PMI
    PMI_pair = PMI(unigrams, up5_bigrams, unigrams_num, bigrams_num)
    top20_PMI = sorted(PMI_pair.items(), key=lambda kv: kv[1], reverse=True)[:20]
    for pair in top20_PMI:
        print(pair)
