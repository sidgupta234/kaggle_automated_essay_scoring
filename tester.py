import numpy as np, pandas as pd, sklearn, matplotlib.pyplot as plt, seaborn as sns
import re
import math
import statistics
import pickle
#from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pickled_model_vocab = pickle.load(open('vocab.pkl', 'rb'))
pickled_model_convention = pickle.load(open('convention.pkl', 'rb'))

dataset = pd.read_csv('data/test.csv')
english_vocab_frequency =  pd.read_csv('vocabulary_data/unigram_freq.csv')

import time
start_time = time.time()

dataset = pd.read_csv('data/test.csv')
english_vocab_frequency =  pd.read_csv('vocabulary_data/unigram_freq.csv')
english_vocab_frequency_dict =  dict(zip(english_vocab_frequency['word'], english_vocab_frequency['count']))
#print(len(english_vocab_frequency_dict))
oov_list = []
total_unique_words = []
#sentence_number = 0

#print(dataset['full_text'][0])

def vocab_clean_dataset(dataset_row):
    #dataset_row["full_text"] = dataset_row["full_text"].lower()
    dataset_row["full_text"] = dataset_row["full_text"].lower()
    dataset_row["full_text"] = re.sub('[\t\n\r]', ' ', dataset_row["full_text"]) #Removing \n, \t, \r from from full_test
    dataset_row["full_text"] = re.sub('[^0-9a-z]', ' ', dataset_row["full_text"]) #Replace all symbols except a-z and 0-9 with spaces
    dataset_row["full_text"] = re.sub('\s{2,}', ' ', dataset_row["full_text"]) #Replacing two or more spaces with single space
    dataset_row['full_text'] = dataset_row["full_text"].strip() #Removing start or end of line spaces from full_test
    return dataset_row['full_text']

def vocab_avg_unique_words(dataset_row):
    dict_ = {}
    words = dataset_row['full_text'].split()
    for word in words:
        dict_[word] = 1
        
    return len(dict_)/len(words)

def vocab_word_freq_score(dataset_row, english_vocab_frequency_dict):
    global oov_list
    global total_unique_words
    freq_score = 0
    #print(dataset_row)
    words = dataset_row['full_text'].split()
    global sentence_number
    
#     if(sentence_number%50 == 0):
#         print("sentencer_number is ", sentence_number)
#     sentence_number+=1
    for word in words:
#         print(word)
        if word in english_vocab_frequency_dict:
            #print(english_vocab_frequency[english_vocab_frequency['word'] == word])
            total_unique_words.append(word)
            freq_score += math.exp(1/english_vocab_frequency_dict[word])
        else:
            #print(word)
            oov_list.append(word)
            #freq_score += 1/math.log(english_vocab_frequency_dict['the'])
            #freq_score += math.log(1/english_vocab_frequency_dict[word])
            #print(word)
            
    return math.log(freq_score)
       
dataset['full_text'] = dataset.apply(vocab_clean_dataset, axis = 1)
dataset['text_length'] = dataset.apply(lambda x: len(x['full_text'].split()), axis = 1)
dataset['avg_word_length'] = dataset.apply(lambda x: statistics.mean([len(i) for i in x['full_text'].split()]), axis = 1)
dataset['avg_unique_words_per_total_words'] = dataset.apply(vocab_avg_unique_words, axis = 1)
dataset['word_frequency_score'] = dataset.apply(lambda x: vocab_word_freq_score(dataset_row = x, english_vocab_frequency_dict = english_vocab_frequency_dict), axis = 1)
#dataset['vocab_label'] = dataset.apply(vocab_label, axis = 1)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

# Linear Regression

lr_vocab = pickled_model_vocab.predict(dataset)
lr_score = lr_vocab.score(X_train, y_train)

# print("Results for Linear Regression with Train Data")
# print(lr_score)

lr_score = lr_vocab.score(X_test, y_test)

# print("Results for Linear Regression with Test Data")
# print(lr_score)

y_train_predict = lr_vocab.predict(X_train)
y_train_predict = [(round(i*2)/2) for i in y_train_predict]
print("rmse", mean_squared_error(y_train, y_train_predict))
# y_train_list = y_train.to_list()

# total = 0
# correct = 0

# for i in range(len(y_train_list)):
#     total += 1
#     #print(y_test[i])
#     if(y_train_list[i] == y_train_predict[i]):
#         correct +=1 
#         #print(y_test_list[i], y_predict[i])

# print(total, correct)
# print("percentage correct values in train are {:.2f}%".format(100*(correct/total)))
# #print(y_predict)

y_predict = lr_vocab.predict(X_test)
y_predict = [(round(i*2)/2) for i in y_predict]
print("rmse", mean_squared_error(y_test, y_predict))
y_test_list = y_test.to_list()

total = 0
correct = 0

for i in range(len(y_test_list)):
    total += 1
    #print(y_test[i])
    if(y_test_list[i] == y_predict[i]):
        correct +=1 
        #print(y_test_list[i], y_predict[i])

print(correct, total)
print("percentage correct values in test are {:.2f}%".format(100*(correct/total)))
pickle.dump(lr_vocab, open('vocab.pkl', 'wb'))

