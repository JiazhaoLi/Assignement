


def train_distribution():
    data_path = './POS.train'
    tag_count_dict = {}
    unigram_count_dict = {}
    bigram_count_dict = {}
    index = 0 
    with open(data_path) as f:
        for line in f:
            index += 1 
            line = "s/<s>" + ' ' + line 
            line = line.strip().split()
            
            for index in range(len(line)):
                line[index] = line[index].split('/')
            print(line)
            
            for index in range(len(line)):
                # unigram distribution
                # count the pos distribution
                pos_tag = line[index][len(line[index])-1]
                if pos_tag not in tag_count_dict:
                    tag_count_dict[pos_tag] = 1
                else:
                    tag_count_dict[pos_tag] += 1 
                # count the word distribution
                word = ''
                for part in line[index][:-1]:
                    word+= part
                if word not in unigram_count_dict:
                    unigram_count_dict[word] = 0
                else:
                    unigram_count_dict[word] += 1
                # bigrams split 
                print(line)
                exit()
                if index < len(line)-1:
                    print(line[index],line[index+1])
                exit()



    print(len(unigram_count_dict))
    print(len(tag_count_dict))
    print(tag_count_dict.keys())

    