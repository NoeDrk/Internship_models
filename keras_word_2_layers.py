#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:38:46 2023

@author: noederijck
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1999)


words_list = []
inputs = []

#Importing word list
with open("fr_en_short_word_list.txt", "r") as file:
    for line in file:
        word_pair = line.strip().split(",")
        words_list.append(word_pair)
        
#Transforming words to one-hot-encoded input/outputs
for word_pair in range(len(words_list)):
    temp_word = np.zeros(len(words_list))
    temp_word[word_pair] = 1
    inputs.append(temp_word)

targets = inputs.copy()
random.shuffle(targets)

#Turns targets and input into tensors
targets = K.constant(np.array(targets))
inputs = K.constant(np.array(inputs))

#Model structure

adam = Adam(learning_rate=0.01)

model = Sequential([
    Dense(units=30, input_shape=(30, ), activation='softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(x=inputs, validation_split=0.1, y=targets, batch_size=10, epochs=30, shuffle=True, verbose=2)



loss = history.history['loss']
accuracy = history.history['accuracy']

#plt.plot(accuracy, color="black")
#plt.plot(loss)
