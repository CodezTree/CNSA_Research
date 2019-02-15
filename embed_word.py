import numpy as np
import re
import tensorflow as tf
import random



####################################################
# cut words function                               #
####################################################
def cut(contents, cut=2):
    results = []
    for content in contents:
        words = content.split()
        result = []
        for word in words:
            result.append(word[:cut])
        results.append(' '.join([token for token in result]))
    return results

####################################################
# divide train/test set function                   #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
def make_input(documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    print(vocab_dict)
    print(len(vocab_dict))
    # Sort the vocabulary dictionary on the basis of values(id).
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

    # vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    # x = np.array(list(vocab_processor.fit_transform(documents)))
    #
    # data = ''
    # for doc in documents:
    #     data += ' '+doc
    # print(data)
    #
    # text_all = list(set(data.split(' ')))
    # print(text_all)
    # vocab_dict = dict(zip(range(len(text_all)), text_all))
    #
    # sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    #
    # vocabulary = list(list(zip(*sorted_vocab))[0])
    #
    # return x, vocabulary, len(vocabulary)

####################################################
# make output function                             #
####################################################
def make_output(points, threshold):
    results = np.zeros((len(points),2))
    for idx, point in enumerate(points):
        if point > threshold:
            results[idx,0] = 1
        else:
            results[idx,1] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path, eng=True, num=True, punc=False):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    # corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    def cut_last(x) :
        return x[:len(x) - 1]

    f = open(data_path, 'rt', encoding='latin-1')
    corpus = f.readlines()
    corpus = list(map(cut_last, corpus))

    corpus = np.array(corpus)
    contents = []
    points = []
    for idx,doc in enumerate(corpus):
        points.append(int(doc[0]))
        contents.append(doc[2:])
        if idx % 1000 == 0 or idx == len(corpus) - 1:
            print('%d docs / %d save' % (idx, len(corpus) - 1))
    return contents, points

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False