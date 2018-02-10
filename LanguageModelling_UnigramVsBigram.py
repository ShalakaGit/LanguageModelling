# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:46:07 2017

@author: Shalaka
"""
import nltk 
from nltk.corpus import udhr  

# Import the DataSet

english = udhr.raw('English-Latin1') 
french = udhr.raw('French_Francais-Latin1') 
italian = udhr.raw('Italian_Italiano-Latin1') 
spanish = udhr.raw('Spanish_Espanol-Latin1')  

# Clearing out the Punctuations and unnecessary characters

from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist, ConditionalFreqDist
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')        
english = tokenizer.tokenize(english)
french = tokenizer.tokenize(french)
italian = tokenizer.tokenize(italian)
spanish = tokenizer.tokenize(spanish)
english_train= english[0:1000]

english_dev = english[1000:1100]
french_train, french_dev = french[0:1000], french[1000:1100] 
italian_train, italian_dev = italian[0:1000], italian[1000:1100] 
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]  
english_test = udhr.words('English-Latin1')[0:1000] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

english_test = ' '.join(english_test)
english_test= tokenizer.tokenize(english_test)

italian_test = ' '.join(italian_test)
italian_test = tokenizer.tokenize(italian_test)

#----------------------------------------------------------------------------------------

#Generating unigram models : I do this by first creating list of character sequences and 
#assigning equal probabilities to them

# Converting the Words to lower cases

english_train_list = list(english_train)
for i in range(len(english_train_list)):
    english_train_list[i] = english_train_list[i].lower()
    english_train[i] = english_train[i].lower()

for i in range(len(french_train)):
    french_train[i] = french_train[i].lower()

for i in range(len(spanish_train)):
    spanish_train[i] = spanish_train[i].lower()

for i in range(len(italian_train)):
    italian_train[i] = italian_train[i]

for i in range(len(english_test)):
    english_test[i] = english_test[i].lower() 
    
for i in range(len(italian_test)):
    italian_test[i] = italian_test[i].lower() 
    
english_train_words = [] 


# Finding Frequency distributions of words on the lists

f_unigram_English = FreqDist(english_train_list)
f_unigram_French = FreqDist(french_train)
f_unigram_Italian = FreqDist(italian_train)
f_unigram_Spanish = FreqDist(spanish_train)


# Unigram Dictionaries to store word : Count pair
unigram_English = {}
unigram_French = {}
unigram_Italian = {}
unigram_Spanish = {}

# Calculating the Frequency distribution Probabilities

for dist in english_train:
    unigram_English[dist] = np.divide(f_unigram_English[dist],1000)
        
for dist in french_train:
    unigram_French[dist] = np.divide(f_unigram_French[dist],1000)

for dist in italian_train:
    unigram_Italian[dist] = np.divide(f_unigram_Italian[dist],1000)

for dist in spanish_train:
    unigram_Spanish[dist] = np.divide(f_unigram_Spanish[dist],1000)

# English Vs French Unigrams

EnglishVsFrench_UnigramsAccuracy = {}
for word in english_test:
    if word in unigram_English:
        i = unigram_English[word]   
    else:
        i = 0       # Assign zero probability if the word's not present 
    if word in unigram_French:
        j = unigram_French[word]
    else:
        j = 0   # Assign zero probability if the word's not present 
    EnglishVsFrench_UnigramsAccuracy[word] = (i,j)  # Creating a dictionary, key: Word, Value : Tuple of it's probabilities in english and french
#print(EnglishVsFrench_UnigramsAccuracy)

correctPredictions_EnglishVsFrench_Unigram = 0      # Number of correct predictions Unigram

# If the English probability in the tuple is greater, count it as correct

for word in EnglishVsFrench_UnigramsAccuracy:
    entry = EnglishVsFrench_UnigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_EnglishVsFrench_Unigram += 1

#-------------------------------#-------------------------------------------------------
SpanishVsItalian_UnigramsAccuracy = {}

for word in italian_test:
    if word in unigram_Italian:
        i = unigram_Italian[word]
    else:
        i = 0
    if word in unigram_Spanish:
        j = unigram_Spanish[word]
    else:
        j = 0
    SpanishVsItalian_UnigramsAccuracy[word] = (i,j)
#print(SpanishVsItalian_UnigramsAccuracy)

correctPredictions_SpanishVsItalian_Unigram = 0

for word in SpanishVsItalian_UnigramsAccuracy:
    entry = SpanishVsItalian_UnigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_SpanishVsItalian_Unigram += 1

#----------------------------------------------------------------------------------

#  Creating English Bigrams dictionary, starting from the second word in the training set, 
# multiply the independent probability of that word by the p(previous word)

English_Bigrams = {}
cnt = 0
prev = 0.0
for i in english_train:
    if cnt == 0:
        English_Bigrams[i] = unigram_English[i]
    else:
        English_Bigrams[i] = np.multiply(unigram_English[i],prev)
    cnt = 1
    prev = unigram_English[i]

#----------------------------------------------------------------------------------

French_Bigrams = {}
cnt = 0
prev = 0.0
for i in french_train:
    if cnt == 0:
        French_Bigrams[i] = unigram_French[i]
    else:
        French_Bigrams[i] = np.multiply(unigram_French[i],prev)
    cnt = 1
    prev = unigram_French[i]

#print(French_Bigrams)
#---------------------------------------------------------------------------------
Spanish_Bigrams = {}
cnt = 0
prev = 0.0
for i in spanish_train:
    if cnt == 0:
        Spanish_Bigrams[i] = unigram_Spanish[i]
    else:
        Spanish_Bigrams[i] = np.multiply(unigram_Spanish[i],prev)
    cnt = 1
    prev = unigram_Spanish[i]

#------------------------------------------------------------------------

Italian_Bigrams = {}
cnt = 0
prev = 0.0
for i in italian_train:
    if cnt == 0:
        Italian_Bigrams[i] = unigram_Italian[i]
    else:
        Italian_Bigrams[i] = np.multiply(unigram_Italian[i],prev)
    cnt = 1
    prev = unigram_Italian[i]

#------------------------------------------------------------------------

EnglishVsFrench_BigramsAccuracy = {}

for word in english_test:
    if word in English_Bigrams:
        i = English_Bigrams[word]
    else:
        i = 0
    if word in French_Bigrams:
        j = French_Bigrams[word]
    else:
        j = 0
    EnglishVsFrench_BigramsAccuracy[word] = (i,j)

correctPredictions_EnglishVsFrench_Bigrams = 0

for word in EnglishVsFrench_BigramsAccuracy:
    entry = EnglishVsFrench_BigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_EnglishVsFrench_Bigrams += 1

#------------------------------------------------------------------------

SpanishVsItalian_BigramsAccuracy = {}

for word in italian_test:
    if word in Italian_Bigrams:
        i = Italian_Bigrams[word]
    else:
        i = 0
    if word in Spanish_Bigrams:
        j = Spanish_Bigrams[word]
    else:
        j = 0
    SpanishVsItalian_BigramsAccuracy[word] = (i,j)
print(SpanishVsItalian_BigramsAccuracy)

correctPredictions_SpanishVsItalian_Bigram = 0

for word in SpanishVsItalian_BigramsAccuracy:
    entry = SpanishVsItalian_BigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_SpanishVsItalian_Bigram += 1

#------------------------------------------------------------------------

#Trigrams are generated multiplying probability of word by the probablities of previous two words
#

English_Trigrams = {}
French_Trigrams = {}
Spanish_Trigrams = {}
Italian_Trigrams = {}

cnt = 0
prev = 0.0
prev1 = 0.0
for i in english_train:
    if cnt == 0:
        English_Trigrams[i] = unigram_English[i]
        cnt = 1
        prev = unigram_English[i]
        continue
    elif cnt == 1:
        English_Trigrams[i] = unigram_English[i]
        prev1 = unigram_English[i]
        cnt = 2
        continue
    else:
        English_Trigrams[i] = np.multiply(unigram_English[i],prev*prev1)
    prev1 = prev
    prev = unigram_English[i]
#------------------------------------------------------------------------
    
cnt = 0
prev = 0.0
prev1 = 0.0
for i in french_train:
    if cnt == 0:
        French_Trigrams[i] = unigram_French[i]
        cnt = 1
        prev = unigram_French[i]
        continue
    elif cnt == 1:
        French_Trigrams[i] = unigram_French[i]
        prev1 = unigram_French[i]
        cnt = 2
        continue
    else:
        French_Trigrams[i] = np.multiply(unigram_French[i],prev*prev1)
    prev1 = prev
    prev = unigram_French[i]

#------------------------------------------------------------------------
    
cnt = 0
prev = 0.0
prev1 = 0.0
for i in italian_train:
    if cnt == 0:
        Italian_Trigrams[i] = unigram_Italian[i]
        cnt = 1
        prev = unigram_Italian[i]
        continue
    elif cnt == 1:
        Italian_Trigrams[i] = unigram_Italian[i]
        prev1 = unigram_Italian[i]
        cnt = 2
        continue
    else:
        Italian_Trigrams[i] = np.multiply(unigram_Italian[i],prev*prev1)
    prev1 = prev
    prev = unigram_Italian[i]

#------------------------------------------------------------------------

cnt = 0
prev = 0.0
prev1 = 0.0
for i in spanish_train:
    if cnt == 0:
        Spanish_Trigrams[i] = unigram_Spanish[i]
        cnt = 1
        prev = unigram_Spanish[i]
        continue
    elif cnt == 1:
        Spanish_Trigrams[i] = unigram_Spanish[i]
        prev1 = unigram_Spanish[i]
        cnt = 2
        continue
    else:
        Spanish_Trigrams[i] = np.multiply(unigram_Spanish[i],prev*prev1)
    prev1 = prev
    prev = unigram_Spanish[i]

#------------------------------------------------------------------------

EnglishVsFrench_TrigramsAccuracy = {}

for word in english_test:
    if word in English_Trigrams:
        i = English_Trigrams[word]
    else:
        i = 0
    if word in French_Trigrams:
        j = French_Trigrams[word]
    else:
        j = 0
    EnglishVsFrench_TrigramsAccuracy[word] = (i,j)

#------------------------------------------------------------------------

correctPredictions_EnglishVsFrench_Trigrams = 0

for word in EnglishVsFrench_TrigramsAccuracy:
    entry = EnglishVsFrench_TrigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_EnglishVsFrench_Trigrams += 1

SpanishVsItalian_TrigramsAccuracy = {}

for word in italian_test:
    if word in Italian_Trigrams:
        i = Italian_Trigrams[word]
    else:
        i = 0
    if word in Spanish_Trigrams:
        j = Spanish_Trigrams[word]
    else:
        j = 0
    SpanishVsItalian_TrigramsAccuracy[word] = (i,j)

#------------------------------------------------------------------------

correctPredictions_SpanishVsItalian_Trigrams = 0

for word in SpanishVsItalian_TrigramsAccuracy:
    entry = SpanishVsItalian_TrigramsAccuracy[word]
    if (entry[0] >= entry[1]):
        correctPredictions_SpanishVsItalian_Trigrams += 1
#------------------------------------------------------------------------

print("Accuracy of Unigram EnglishVsFrench: ",np.divide(correctPredictions_EnglishVsFrench_Unigram,1000))
print("Accuracy of Bigram EnglishVsFrench: ",np.divide(correctPredictions_EnglishVsFrench_Bigrams,1000))
print("Accuracy of Trigram EnglishVsFrench: ",np.divide(correctPredictions_EnglishVsFrench_Trigrams,1000))

print("Accuracy of Unigram SpanishVsItalian: ",np.divide(correctPredictions_SpanishVsItalian_Unigram,1000))
print("Accuracy of Bigram SpanishVsItalian: ",np.divide(correctPredictions_SpanishVsItalian_Bigram,1000))
print("Accuracy of Trigram SpanishVsItalian: ",np.divide(correctPredictions_SpanishVsItalian_Trigrams,1000))

#------------------------------------------------------------------------
