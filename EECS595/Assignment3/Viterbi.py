
from util import *

train_sample, tag_word, tag_nexttag, tag_dict, word_dict = train_load()

test_sample = test_load()

predict_sen_samples, true_sen_samples = Viterbi_model(test_sample,tag_word,tag_nexttag, tag_dict, word_dict)
tag_acc, sen_acc  = predict(predict_sen_samples,true_sen_samples)
print('Viterbi_tag_level acc = ' + str(tag_acc))
print('Viterbi_sen_level acc= '+ str(sen_acc))
ouput(predict_sen_samples,tag_dict)

baseline_acc = simple_baseline_predict(tag_word, tag_nexttag, tag_dict, word_dict)

print('baseline_acc = '+ str(baseline_acc))