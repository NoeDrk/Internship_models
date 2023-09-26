#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:50:58 2023

@author: noederijck
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:38:46 2023

@author: noederijck
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import random
import math
import matplotlib.pyplot as plt

random.seed(1999)
words_list = []

alphabet = {"a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "i": 8,
            "j": 9,
            "k": 10,
            "l": 11,
            "m": 12,
            "n": 13,
            "o": 14,
            "p": 15,
            "q": 16,
            "r": 17,
            "s": 18,
            "t": 19,
            "u": 20,
            "v": 21,
            "w": 22,
            "x": 23,
            "y": 24,
            "z": 25}

fr_labels = []
en_train = []



#Import fr/en word pairs
with open("fr_en_long_word_list.txt", "r") as file:
    for line in file:
        word_pair = line.strip().split(",")
        words_list.append(word_pair)
                                                      
# Transforming words to input vector of lenght 26 based on the frequency of each letter in the word
for words in words_list:
    en_word = np.zeros(26)
    fr_word = np.zeros(26)
    for letter in words[0]:
        en_word[alphabet[letter]] += 1/len(words[0])
    for letter in words[1]:
        fr_word[alphabet[letter]] += 1/len(words[1])
    en_train.append(en_word)
    fr_labels.append(fr_word)
#                                 A,    B,   C,   D,  ...Z
# Such that the word DAD becomes [0.33, 0.0, 0.0, 0.66...0.0]

# Because 2/3 of the letters in the word "DAD" are "D" and "A" represents 1/3 of the letters


# Turns list of list into matrix of tensors
en_train = tf.stack(np.array(en_train))
fr_labels = tf.stack(np.array(fr_labels))

#Model Parameters
optimizer = Adam(learning_rate=0.001)

#Model Structure
model = Sequential([Dense(units=50, input_shape=(26,), activation='relu')])
model.add(Dense(units=26, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x=en_train, validation_split=0.1, y=fr_labels, batch_size=30, epochs=200, shuffle=True, verbose=2)


#loss = history.history['loss']
#accuracy = history.history['accuracy']
#plt.scatter(loss, accuracy, marker='o', c='black')
#plt.title('Accuracy vs. Loss')

plt.plot(history.history["loss"], color = "black")
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')    
#plt.show()
    
    
    
