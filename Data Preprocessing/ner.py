#necessary library
import nltk
import pandas as pd
import numpy as np
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

#read csv file and separate input and output
df = pd.read_csv('qd.csv', header=None)
x = df.iloc[:,:1].values
y = df.iloc[:,1:].values

'''temp_input = []
for i in range(x.shape[0]):
    temp = nltk.word_tokenize(x[i][0])
    temp_input.append(temp)

print(temp_input)'''

#get the Name Entity Labelling
totalInput = []
for i in range(len(x)):
    doc = nlp(x[i][0])
    temp1 = [X.label_ for X in doc.ents]
    temp2 = nltk.word_tokenize(x[i][0])
    iP = temp2+temp1
    print(iP)
    totalInput.append(iP)

#List of vocabulary in the dataset
vocab = []
for i in totalInput:
    for j in i:
        vocab.append(j)
print('vocab', len(vocab))

#get 2000 most frequent word count
from nltk import FreqDist
fdist = FreqDist(vocab)
temp5 = fdist.most_common(2000)
print('mostFreq' ,temp5)

#make tokens of at least 8 times repeated words
most_common_token = []
for i in temp5:
    if i[1]>=8:
        most_common_token.append(i[0])
print(len(most_common_token))


#binarize the input using bag-of-words
input_bin = []
len_s  = len(most_common_token)
for i in totalInput:
    temp = [0] * len_s
    for j in i:
        if j in most_common_token:
            temp[most_common_token.index(j)] = 1
    input_bin.append(temp)


#save input into csv file
np_input = np.asarray(input_bin)
print(np_input.shape)
np.savetxt("QD_BIN_FINAL.csv", np_input, delimiter=",")
print('----------------saved-----------------')











