import nltk
import matplotlib.pyplot as plt
import numpy as np
import string
from nltk import pos_tag

def stop_word(filename):
	with open(filename) as f:
		stopword = dict()
		for line in f:
			line = line.strip('\n')
			words = line.split()
			for word in words:
				word = word.lower()
				if word not in stopword:
					stopword[word] = 1
				else:
					stopword[word] += 1
	return stopword

def DataLoad(filename):
	with open(filename) as f:
		word_dict = dict()
		word_num = 0
		for line in f:
			line = line.strip('\n')
			words = nltk.word_tokenize(line)
			for word in words:
				if word not in string.punctuation:
					word = word.lower()
					word_num += 1
					if word not in word_dict:
						word_dict[word] = 1
					else:
						word_dict[word] += 1
	return word_dict, word_num


def Distribution_Word(WordDict, WordNum, stop_word, filename):
	Degree_Num = dict()
	for word, num in WordDict.items():
		if word not in stop_word:
			if num not in Degree_Num:
				Degree_Num[num] = 1
			else:
				Degree_Num[num] += 1

	sortList_degree = sorted(Degree_Num.items(), key=lambda kv: kv[0])
	fre_list = []
	degree_list = []
	for pair in sortList_degree:
		degree_list.append(np.log(pair[0]))
		fre_list.append(np.log(float(pair[1])/WordNum))

	plt.plot(degree_list, fre_list, 'o')
	plt.title(filename+"_word_distribution")
	plt.xlabel("Degrees of word(times(log)")
	plt.ylabel("Frequence of word(log)")
	plt.savefig(filename+'Word_distribution.png')
	plt.show()


def get_proportion_stopword(word_num, word_dict, stop_word):
	stop_num = 0
	for word, num in word_dict.items():
		# a frequence of stopword
		if word in stop_word:
			stop_num += num
	print("frequency of stopword: " + str(float(stop_num)/word_num))


def get_cap_letters_lenword(filename):
	letter_num = 0
	cap_num = 0
	word_num = 0
	word_total_len = 0
	with open(filename) as f:
		for line in f:
			line = line.strip('\n')
			words = nltk.word_tokenize(line)
			for word in words:
				if word not in string.punctuation:
					word_num += 1
					word_total_len += len(word)
					for letter in word:
						letter_num += 1
						if letter.isupper():
							cap_num += 1
	print(filename + ': average of len of word = '+ str(float(word_total_len)/word_num))
	print(filename + ': proportion of cap letters = '+ str(float(cap_num)/letter_num))


def get_tag(filename,stopword):
	with open(filename) as f:
		pos_dict = dict()
		noun_dict = dict()
		verb_dict = dict()
		adj_dict = dict()
		tag_list = ['NOUN', "VERB",'ADJ','ADV']
		# percentage of noun, adj, verb, adverb, pornouns
		for line in f:
			line = line.strip('\n')
			words = nltk.word_tokenize(line)
			for word in words:
				if word in string.punctuation:
					words.remove(word)
			for word in words:
				if word in stopword:
					words.remove(word)
			tag_pair = nltk.pos_tag(words, tagset='universal')
			for pair in tag_pair:

				if pair[1] not in pos_dict:
					pos_dict[pair[1]] = 1
				else:
					pos_dict[pair[1]] += 1
				if pair[1] == 'NOUN':
					if pair[0] not in noun_dict:
						noun_dict[pair[0]] = 1
					else:
						noun_dict[pair[0]] += 1
				if pair[1] == 'VERB':
					if pair[0] not in verb_dict:
						verb_dict[pair[0]] = 1
					else:
						verb_dict[pair[0]] += 1
				if pair[1] == 'ADJ':
					if pair[0] not in adj_dict:
						adj_dict[pair[0]] = 1
					else:
						adj_dict[pair[0]] += 1

    
		return [pos_dict, noun_dict, verb_dict, adj_dict]


def get_tag_distribution(filename, word_num,stop_word):
	taglist = get_tag(filename,stop_word)
	pos_dict = taglist[0]
	noun_dict = taglist[1]
	verb_dict = taglist[2]
	adj_dict = taglist[3]
	for tag, num in pos_dict.items():
		print(tag + ": " + str(float(num) / word_num))
	sort_noun = [pair[0] for pair in sorted(noun_dict.items(), key=lambda kv: kv[1], reverse=True)]
	print(sort_noun[:10])
	sort_verb = [pair[0] for pair in sorted(verb_dict.items(), key=lambda kv: kv[1], reverse=True)]
	print(sort_verb[:10])
	sort_adj = [pair[0] for pair in sorted(adj_dict.items(), key=lambda kv: kv[1], reverse=True)]
	print(sort_adj[:10])


def analysis(filename, word_dict,word_num, stop_word ):
	print(filename)
	# a
	get_proportion_stopword(word_num, word_dict, stop_word)
	#b, c
	get_cap_letters_lenword(filename)
	# de
	get_tag_distribution(filename, word_num, stop_word)


def TFIDF(collocation,stop_word):
	N = 0
	with open(collocation) as f:
		doc_index = 0
		tuple_list = list()
		for doc in f:
			doc_len = 0
			N+=1
			word_dict = dict()
			doc = doc.strip('\n')
			words = nltk.word_tokenize(doc)
			for word in words:
				word = word.lower()
				if word not in string.punctuation and word not in word_dict and word not in stop_word:
					word_dict[word] = 1
					doc_len += 1
				elif word not in string.punctuation and word not in stop_word:
					word_dict[word] += 1
					doc_len += 1
			for word, count in word_dict.items():
				tuple_list.append((doc_index, word, np.log(float(count) + 1)))
			doc_index += 1

	word_sort = sorted(tuple_list, key=lambda pair: pair[1])
	word_doc = dict()


	for pair in word_sort:
		if pair[1] not in word_doc:
			doc_list = []
			doc_list.append(pair[0])
			word_doc[pair[1]] = doc_list
		else:
			doc_list = word_doc[pair[1]]
			doc_list.append(pair[0])
			word_doc[pair[1]] = doc_list
	# this is IDF
	for word, doc_list in word_doc.items():
		doc_len = len(set(doc_list))
		word_doc[word] = 1 + np.log(float(N)/doc_len)


	TF_IDF_doc = dict()
	for doc_index in range(10):
		TF_IDF_list = list()
		for pair in tuple_list:
			if pair[0] == doc_index:
				TF_IDF_list.append((pair[0], pair[1], pair[2] * word_doc[pair[1]]))
		TF_IDF_doc[doc_index] = TF_IDF_list

	for doc, tuple in TF_IDF_doc.items():
		print('doc' + str(doc))
		top5_list = sorted(tuple, key=lambda x: x[2], reverse=True)[:5]
		print([pair[1] for pair in top5_list])


if __name__ == "__main__":

	stop_word = stop_word('stoplist.txt')
	ehr_word_dict, ehr_word_num = DataLoad('ehr.txt')
	medhelp_word_dict, medhelp_word_num = DataLoad('medhelp.txt')

	Distribution_Word(medhelp_word_dict, medhelp_word_num, stop_word, 'medhelp.txt')
	Distribution_Word(ehr_word_dict, ehr_word_num, stop_word, 'whr.txt')

	analysis('ehr.txt', ehr_word_dict, ehr_word_num, stop_word)
	analysis('medhelp.txt', medhelp_word_dict, medhelp_word_num, stop_word)


	TFIDF('ehr.txt',stop_word)



