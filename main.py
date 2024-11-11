# This is a sample Python script.
#import modules
import os.path
import re

# from joblib.numpy_pickle_utils import xrange
# from gensim import corpora
# from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from collections import Counter
import math


# from gensim.models.coherencemodel import CoherenceModel
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def load_data(path,file_name):

    with open( os.path.join(path, file_name) ,"r") as file:
        FileOne = file.read()

    return FileOne

def preprocess_data(train_text):

    train_text = train_text.replace('.', '')
    train_text = train_text.replace(',', '')
    train_text = train_text.replace('``', '')
    # STOP WORDS
    stop_words = set(stopwords.words('english'))
    # TOKENIZE
    word_tokens = word_tokenize(train_text)
    filtered_content = []
    # STEMMING
    porter = PorterStemmer()
    for w in word_tokens:
        if w not in stop_words:
            w = w.lower()
            word = porter.stem(w)
            filtered_content.append(word)

    return filtered_content

def tf(question, word):
    if word not in question:
        return 0
    count = dict(Counter(question))
    q_len = len(question)
    return float(count[word]) / float(q_len)

def n_containing(qlist, word):
    return float(qlist[word])

def idf(qlist, word):
    return math.log(float(len(qlist.keys())) / (1.0 + n_containing(qlist, word)))

def tfidf(question, qlist, word):
    return tf(question, word) * idf(qlist, word)

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])

#calculate Levenshtein distance between two arrays
def levenshtein_for_text(fileOne,fileTwo):
    c=0
    equal=[]
    for i in fileOne:
        for k in fileTwo:
            if levenshtein(i, k)<1:
                equal.append(k)
                # print(i,k,"distance: ",levenshtein(i, k))

    # print(set(equal),len(set(equal))/len(k))
    return equal


def detect(paths, f1, f2):
    fOne = preprocess_data(load_data(paths, f1))
    ftwo = preprocess_data(load_data(paths, f2))
    qlist = []
    qlist += fOne + ftwo
    qlist = dict(Counter(qlist))
    TOne = []
    sumone = 0
    sumtwo = 0
    for i in set(levenshtein_for_text(preprocess_data(load_data(paths, f1)), preprocess_data(load_data(paths, f2)))):
        # print("word: ",i,"tf: ", tf(fOne, i))
        # print("word: ", i, "tfidf: ", tfidf(fOne, qlist, i))
        sumone = (1 / tfidf(fOne, qlist, i)) + sumone
        # print("word: ", i, "tfidf: ", tfidf(ftwo, qlist, i))
        sumtwo = (1 / tfidf(ftwo, qlist, i)) + sumtwo

    drateone = sumone / (len(preprocess_data(load_data(paths, f1))) * 100)
    dratetwo = sumtwo / (len(preprocess_data(load_data(paths, f2))) * 100)
    # print(drateone,dratetwo)
    result = max(drateone,dratetwo)
    return result


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')


f1="1.txt"
f2="2.txt"
for i in range(1,8):
    print("detected rate: ",i,detect("/Users/mojdeh/PycharmProjects/ADT/data/okay0"+str(i),f1,f2))

# paths="/Users/mojdeh/PycharmProjects/ADT/data/plagiarism01"










# tfidf(question, qlist, word)
print("preprocessData File one:",preprocess_data(load_data(paths,f1)),"preprocessData File Two:", '\n', preprocess_data(load_data(paths,f2)))

levenshtein_for_text(preprocess_data(load_data(paths,f2)),preprocess_data(load_data(paths,f2)))

