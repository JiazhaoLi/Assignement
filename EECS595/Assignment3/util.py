
import numpy as np
import sys
def form_train_distribution():
    # data_path = './POS.train'
    data_path = sys.argv[1]
    tag_dict = {}
    tag_index = 0
    word_dict = {}
    word_index = 0
    with open(data_path) as f:
        for line in f:
            line = "<s>/<s>" + ' ' + line 
            line = line.strip().split()
            new_line = []
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)

            for index in range(len(new_line)):
                word = new_line[index][0]
                pos_tag = new_line[index][1]
                if word not in word_dict:
                    word_dict[word] = word_index
                    word_index += 1
                if pos_tag not in tag_dict:
                    tag_dict[pos_tag] = tag_index
                    tag_index += 1 
    
    word_dict['NON'] = word_index
    num_tag = len(tag_dict)
    num_word = len(word_dict)
    
    tag_nexttag = np.ones((num_tag, num_tag))
    tag_word = np.ones((num_tag, num_word))
    with open(data_path) as f:
        for line in f:
            line = "<s>/<s>" + ' ' + line 
            line = line.strip().split()
            new_line = []
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            for index in range(len(new_line)):
                word = new_line[index][0]
                pos_tag = new_line[index][1]
                tag_word[tag_dict[pos_tag],word_dict[word]] += 1
                if index+1 < len(new_line):
                    next_tag = new_line[index+1][1]
                    tag_nexttag[tag_dict[pos_tag], tag_dict[next_tag]] += 1 

    tag_word = (tag_word/tag_word.sum(axis=1, keepdims=1)) 
    tag_nexttag = tag_nexttag/tag_nexttag.sum(axis=1, keepdims=1) 

    return tag_word, tag_nexttag, tag_dict, word_dict

def train_load():
    tag_word, tag_nexttag,tag_dict, word_dict = form_train_distribution()
    # data_path = './POS.train'
    data_path = sys.argv[1]
    train_sample = []
    with open(data_path) as f:
        for line in f:
            line = "<s>/<s>" + ' ' + line 
            line = line.strip().split()
            new_line = []
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            train_sample.append(new_line)
    # print(tag_word)
    # print(tag_nexttag)
    return train_sample,tag_word,tag_nexttag, tag_dict, word_dict

def test_load():
    data_path = sys.argv[2]
    test_sample = []
    with open(data_path) as f:
        for line in f:
            line = "<s>/<s>" + ' ' + line 
            line = line.strip().split()
            new_line = []
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            test_sample.append(new_line)
    return test_sample
    
def Viterbi_model(test_sample,tag_word,tag_nexttag, tag_dict, word_dict):

    predict_sen_samples = []
    true_sen_samples = []
    for instance in test_sample:
        word_score = np.zeros((len(tag_dict),len(instance)))
        trace_back = np.zeros((len(tag_dict),len(instance)))
        label_list = np.zeros(len(instance))
         # initial part 
        tag_init = '<s>'
        word = instance[1][0]
        if word not in word_dict:
            word = 'NON'
        label_list[1] = (tag_dict[instance[1][1]])
        word_tag_prob_list = tag_word[:, word_dict[word]]
        tag_nexttag_prob_list = tag_nexttag[0,:]
        score_list = list(word_tag_prob_list * tag_nexttag_prob_list)
       
        word_score[:,1] = score_list  # list 
        trace_back[:,1] = np.zeros((len(tag_dict))) # index 
        # iteration to the end
        for i in range(2,len(instance)):
            score_list = []# max score for each one 
            back_list = []   # max score postion
            word = instance[i][0]  
            if word not in word_dict:
                word = 'NON'
            label_list[i] = (tag_dict[instance[i][1]]) 
            # tag_label = instance[i][1]   
            previous_score = np.array(word_score[:, i-1]).reshape(-1, 1)
            transition_score = previous_score * tag_nexttag 
            for j in range(len(tag_dict)):  # for each tag 
                word_pos_prob = tag_word[j, word_dict[word]]
                score_list.append(max(transition_score[:, j] * word_pos_prob))   # socro for this tag 
                back_list.append(np.argmax(transition_score[:, j] * word_pos_prob))
            
            word_score[:, i] = np.array(score_list)
            trace_back[:, i] = np.array(back_list)
        predict_label = np.zeros((len(instance)))
        final_maxone = np.argmax(word_score[:,-1])
        
        predict_label[-1] = int(final_maxone)
        for i in range(len(instance)-2,0,-1):
            final_maxone = int(trace_back[final_maxone,i+1])
            predict_label[i] = int(final_maxone)


        predict_sen_samples.append(predict_label)
        true_sen_samples.append(label_list)
    
    return predict_sen_samples, true_sen_samples


def predict(predict_sen_samples,true_sen_samples):
    sen_count = 0
    tag_count = 0
    sentence_correct = 0 
    tag_wrong = 0
    for sen in range(len(predict_sen_samples)):
        sen_count += 1 
        flag = 1
        for tag in range(len(predict_sen_samples[sen])):
            tag_count += 1
            if (predict_sen_samples[sen][tag]) != (true_sen_samples[sen][tag]):
                flag = 0
                tag_wrong += 1 
        if flag == 1:
            sentence_correct += 1
            

    # print(tag_count)
    # print(sen_count)
    # print(tag_wrong)
    tag_acc = (tag_count - tag_wrong) / tag_count
    sen_acc = sentence_correct/sen_count

    return tag_acc, sen_acc
       
def ouput(predict_sen_samples,tag_dict):
    index_tag_dict={}
    for tag,index in tag_dict.items():
        index_tag_dict[index] = tag
    data_path = sys.argv[2]

    test_sample = []
    index = 0 
    with open(data_path) as f:
        for line in f:
            line = line.strip().split()
            new_line = ''
            item_index = 0
            for pair in line:
                pair = pair.split('/')
                word = pair[0]
                if len(pair) == 2:
                    tag = pair[1]
                    new_line += word+'/'+index_tag_dict[predict_sen_samples[index][item_index+1]]+' '
                    item_index += 1
            index+=1
            test_sample.append(new_line)
    with open('POS.test.out','w') as f :
        for sentenc in test_sample:
            f.write(sentenc)
            f.write('\n')


def simple_baseline_predict(tag_word, tag_nexttag, tag_dict, word_dict):
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    tag_count = 0
    acc_count = 0
    error_anal = 0
    num_tag = len(tag_dict)
    num_word = len(word_dict)
    word_tag = np.ones((num_word, num_tag))

    with open(train_data_path) as f:
        for line in f:
            line = line.strip().split()
            new_line = []
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            for index in range(len(new_line)):
                word = new_line[index][0]
                pos_tag = new_line[index][1]
                if word not in word_dict:
                    word = 'NON'
                word_tag[word_dict[word], tag_dict[pos_tag]] += 1
    word_tag = word_tag /word_tag.sum(axis=1, keepdims=1)

    with open(test_data_path) as f:
        for line in f:
            line = line.strip().split()
            for pair in line:
                pair = pair.split('/')
                if len(pair) == 2:
                    tag_count += 1
                    word = ''
                    if pair[0] not in word_dict:
                        word = 'NON'
                    else:
                        word = pair[0]
                    predict_tag = np.argmax(word_tag[word_dict[word], :])
                    if predict_tag == tag_dict[pair[1]]:
                        acc_count += 1
    return acc_count / tag_count
    

          
            
            
            
        
    