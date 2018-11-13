import numpy as np
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import csv
import pdb
import nltk

train_path = './all/train.tsv'
test_path = './all/test.tsv'
output_path = './all/submission.csv'
wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()

def loadTrainingData(file_path):
    print('loading data ......')
    label = []
    comments = []
    parent_comments = []
    stopword = nltk.corpus.stopwords.words('english')
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            com = []
            par_com = []
            line = line.split('\t')
            label.append(int(line[0]))
            raw_token_comment = [re.split(r'[.,!?;()#$%&*+,-./:<=>@[\]^_`{|}~\d"]', x) for x in line[1].lower().split(' ')]
            for t in raw_token_comment:
                for item in t:
                    if item != '': #and len(item) >= 2 and len(item) <= 12:
                        #lemma_word = wordnet_lemmatizer.lemmatize(item)
                        #stem_word = lancaster_stemmer.stem(lemma_word)
                        #if lemma_word not in stopword:
                        com.append(item)
            comments.append(com)
            raw_token_par_comment = [re.split(r'[.,!?;()#$%&*+,-./:<=>@[\]^_`{|}~\d"]', x) for x in line[2].lower().split(' ')]
            for t in raw_token_par_comment:
                for item in t:
                    if item != '': #and len(item) >= 2 and len(item) <= 12:
                        #lemma_word = wordnet_lemmatizer.lemmatize(item)
                        #stem_word = lancaster_stemmer.stem(lemma_word)
                        #if lemma_word not in stopword:
                        par_com.append(item)
            parent_comments.append(par_com)
    print('complete data loading !!!')
    return label, comments, parent_comments

def loadTestingData(file_path):
    print('loading testing data ......')
    comments = []
    parent_comments = []
    stopword = nltk.corpus.stopwords.words('english') 
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            com = []
            par_com = []
            line = line.split('\t')
            raw_token_comment = [re.split(r'[.,!?;()#$%&*+,-./:<=>@[\]^_`{|}~\d"]', x) for x in line[1].lower().split(' ')]
            for t in raw_token_comment:
                for item in t:
                    if item != '': #and len(item) >= 2:
                        #lemma_word = wordnet_lemmatizer.lemmatize(item)
                        #stem_word = lancaster_stemmer.stem(lemma_word)
                        #if lemma_word not in stopword:
                        com.append(item)
            comments.append(com)
            raw_token_par_comment = [re.split(r'[.,!?;()#$%&*+,-./:<=>@[\]^_`{|}~\d"]', x) for x in line[2].lower().split(' ')]
            for t in raw_token_par_comment:
                for item in t:
                    if item != '': #and len(item) >= 2:
                        #lemma_word = wordnet_lemmatizer.lemmatize(item)
                        #stem_word = lancaster_stemmer.stem(lemma_word)
                        #if lemma_word not in stopword:
                        par_com.append(item)
            parent_comments.append(par_com)
    print('complete data loading !!!')
    return comments, parent_comments

def loadData(file_path):
    train_comments = []
    label = []
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            label.append(int(line.strip().split('\t')[0]))
            line = line.strip().split('\t')[1]
            train_comments.append(line)
    return train_comments, label

def wordCountFeature(train_comments, test_comments, train_parent_comments, test_parent_comments, label):
    print('embedding word ......')
    idx = len(label)
    train_comments = train_comments + test_comments
    train_parent_comments = train_parent_comments + test_parent_comments
    train_comments_new = [' '.join(x) for x in train_comments]
    train_parent_comments_new = [' '.join(x) for x in train_parent_comments]
    vectorizer = CountVectorizer()
    X_comments = vectorizer.fit_transform(train_comments_new)
    X_parent_comments = vectorizer.fit_transform(train_parent_comments_new)
    train_X = np.concatenate((X_comments.toarray(), X_parent_comments.toarray()), axis=1)
    train_X, test_X = train_X[:idx,:], train_X[idx:,:] 
    print(len(label))
    print(train_X.shape)
    print(test_X.shape)
    print('count feature is completed !!!')
    return train_X, test_X, np.array(label).reshape(-1)

def TFIDFEmbedding(train_comments, test_comments, label):
    tfidf = TFIDF(min_df=1,
           strip_accents='unicode',
           analyzer='word',
           ngram_range=(2, 3),
           token_pattern=r'\w{1,}',  
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english')

    print('embedding word ......')
    idx = len(label)
    #all_train = [' '.join(train_comments[i]) for i in range(len(train_comments))]
    #all_test = [''.join(test_comments[i]) for i in range(len(test_comments))]
    #all_train = [' '.join(train_parent_comments[i] + train_comments[i]) for i in range(len(train_comments))]
    #all_test = [''.join(test_parent_comments[i] + test_comments[i]) for i in range(len(test_comments))]
    all_data = train_comments + test_comments
    tfidf.fit(all_data)
    all_data = tfidf.transform(all_data)
    '''
    feature_names = tfidf.get_feature_names()
    tfidf_matrix = all_data
    dense = tfidf_matrix.todense()
    episode = dense[1].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:100]:
         print('{0: <100} {1}'.format(phrase, score))
    '''
    train_X = all_data[:idx]
    test_X = all_data[idx:]
    print('TFIDF is completed !!!')
    train_Y = np.array(label).reshape(-1) 
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size = 0.3, random_state=86)
    print('training data size is:',train_X.shape)
    print('testing data size is:', test_X.shape)
    return train_X, val_X, train_Y, val_Y, test_X 

def wordEmbedding(train_comments, test_comments, train_parent_comments, test_parent_comments, label):
    print('embedding wording ......')
    train_comments_list = []
    test_comments_list = []
    train_parent_comments_list = []
    test_parent_comments_list = []
    comments_inital = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_comments + test_comments)]
    parent_comments_inital = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_parent_comments + test_parent_comments)]
    comments_model = Doc2Vec(comments_inital, vector_size=256, window=2, min_count=0, workers=4)
    parent_comments_model = Doc2Vec(parent_comments_inital, vector_size=256, window=2, min_count=0, workers=4)
    
    cnt = 0
    cnt_all = 0
    for item in tqdm(train_comments):
        vector = comments_model.infer_vector(item)
        train_comments_list.append(vector)
    '''
        item_most_similar_doc = comments_model.docvecs.most_similar([vector], topn = 10)
        idx_list = [item[0] for item in item_most_similar_doc]
        label_list = [label[train_comments.index(item)]]
        try:
            cnt_all += 1
            label_list += [label[i] for i in idx_list]
            acc = (label_list.count(label_list[0]) - 1)/10
            if acc > 0.7:
                cnt += 1
            #print(label_list, acc, cnt)
        except IndexError:
            continue
    print(cnt, cnt_all)
    '''
    for item in tqdm(train_parent_comments):
        vecotr = parent_comments_model.infer_vector(item)
        train_parent_comments_list.append(vecotr)
    for item in tqdm(test_comments):
        vector = comments_model.infer_vector(item)
        test_comments_list.append(vector)
    for item in tqdm(test_parent_comments):
        vecotr = parent_comments_model.infer_vector(item)
        test_parent_comments_list.append(vecotr)

    print('complete data embedding !!!')
    return train_comments_list, test_comments_list, train_parent_comments_list, test_parent_comments_list

def NBModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training NBModel ......')
    model_NB = MNB()
    model_NB.fit(train_X, train_Y)
    MNB(alpha=1.0, class_prior=None, fit_prior=True)
    #print("cross validation score is: ", np.mean(cross_val_score(model_NB, train_X, train_Y, cv=10, scoring='roc_auc')))
    val_preds = np.array(model_NB.predict(val_X))
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_NB.score(val_X, val_Y))
    preds = np.array(model_NB.predict(test_X))
    return preds

def LinearModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training linear regression model ......')
    model_LR = LinearRegression()
    model_LR.fit(train_X, train_Y)
    val_preds = np.array(model_LR.predict(val_X))
    val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds]) 
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    preds = np.array(model_LR.predict(test_X))
    preds = np.array([1 if x > 0.5 else 0 for x in preds])
    return preds

def LogisticeModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training logistic regression model ......')
    model_LR = LR(penalty='l2')
    model_LR.fit(train_X, train_Y)
    val_preds = np.array(model_LR.predict(val_X))
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    #print(model_LR.score(val_X, val_Y))
    preds = np.array(model_LR.predict(test_X))
    return preds

def SVMModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training SVM model ......')
    model_SVM = svm.SVC()
    model_SVM.fit(train_X, train_Y)
    val_preds = np.array(model_SVM.predict(val_X)) 
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    preds = np.array(model_SVM.predict(test_X))
    return preds 

def DecisionTreeModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training DTC model ......')
    model_Tree = DecisionTreeClassifier()
    model_Tree.fit(train_X, train_Y)
    val_preds = np.array(model_Tree.predict(val_X)) 
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    preds = np.array(model_Tree.predict(test_X))
    return preds   

def RandomForestModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training RF model ......')
    model_Forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)
    model_Forest.fit(train_X, train_Y)
    val_preds = np.array(model_Forest.predict(val_X)) 
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary'))
    preds = np.array(model_Forest.predict(test_X))
    return preds  
    
def XgbModel(train_X, val_X, train_Y, val_Y, test_X):
    print('training xgboost model ......')
    num_rounds = 1000
    params1={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth':3, 
        'lambda':1,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'min_child_weight':3, 
        'silent':1,
        'eta': 0.2, 
        'seed':1000,
        #'nthread':7,
        }
    params2={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'gamma':0.1,
        'max_depth':5, 
        'lambda':2,
        'subsample':0.5,
        'colsample_bytree':0.8,
        'min_child_weight':3, 
        'silent':0 ,
        'eta': 0.01, 
        'seed':1000,
        #'nthread':7,
        }
    plst1 = list(params1.items())
    plst2 = list(params2.items())
    xgb_train = xgb.DMatrix(train_X, label=train_Y)
    xgb_val = xgb.DMatrix(val_X,label=val_Y)
    xgb_test = xgb.DMatrix(test_X)
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst1, xgb_train, num_rounds, watchlist)
    val_preds = model.predict(xgb_val, ntree_limit=model.best_ntree_limit) 
    val_preds = np.array([1 if x > 0.5 else 0 for x in val_preds])
    print(len(val_Y), val_Y.tolist().count(1))
    print('Precision score is:', precision_score(val_Y, val_preds, average='binary'))
    print('Recall score is:', recall_score(val_Y, val_preds, average='binary'))
    print('F1 score is:', f1_score(val_Y, val_preds, average='binary')) 
    preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    preds = np.array([1 if x > 0.5 else 0 for x in preds])
    print('training is completed !!!')
    return preds

def fileWritting(output_path, preds):
    print('writting file ...')
    with open(output_path, 'w') as f:
        f = csv.writer(f, delimiter = ',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f.writerow(['Id', 'Category'])
        for i in tqdm(range(len(preds))):
            Id = str(i)
            Category = str(preds[i]) 
            f.writerow([Id, Category])
    print('writting is completed !!!')
            

if __name__ == '__main__':
    #label, train_comments, train_parent_comments = loadTrainingData(train_path)
    #test_comments, test_parent_comments = loadTestingData(test_path)
    train_comments, label = loadData(train_path)
    test_comments, _ = loadData(test_path)
    train_X, val_X, train_Y, val_Y, test_X = TFIDFEmbedding(train_comments, test_comments, label)
    #pdb.set_trace()
    preds_LR = LogisticeModel(train_X, val_X, train_Y, val_Y, test_X)
    print(preds_LR.tolist().count(0), preds_LR.tolist().count(1)) 
    fileWritting(output_path, preds_LR)
    preds_NB = NBModel(train_X, val_X, train_Y, val_Y, test_X)
    print(preds_NB.tolist().count(0), preds_NB.tolist().count(1))
    fileWritting(output_path, preds_NB)
    preds_Linear = LinearModel(train_X, val_X, train_Y, val_Y, test_X)
    print(preds_Linear.tolist().count(0), preds_Linear.tolist().count(1)) 
    fileWritting(output_path, preds_Linear) 
    '''
    preds_SVM = SVMModel(train_X, val_X, train_Y, val_Y, test_X)
    preds_DTC = DecisionTreeModel(train_X, val_X, train_Y, val_Y, test_X)
    prdes_RF = RandomForestModel(train_X, val_X, train_Y, val_Y, test_X)
    preds_XGB = XgbModel(train_X, val_X, train_Y, val_Y, test_X)
    #fileWritting(output_path, preds)
    '''