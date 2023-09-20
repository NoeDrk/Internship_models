#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



random.seed(1999)

words_list = []
inputs = []
epoch_accuracies = []
epoch_losses = []


#Importing word list
with open("/Users/noederijck/Desktop/word_lists/fr_en_short_word_list.txt", "r") as file:
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

#Initializing biases and weights
targets = np.array(targets)
inputs = np.array(inputs)

weights = np.random.rand(30, 30)
biases = np.zeros(30)

#Model parameters
learning_rate = 0.1
epochs = 500


#Model Training
for epoch in range(epochs):
    inputs, targets = shuffle(inputs, targets)
    total_loss = 0
    correct_predictions = 0
    for n in range(30):
        output_activation = np.dot(inputs[n], weights) + biases
        
        #Softmax activation function
        exp_values = np.exp(output_activation)
        norm_values = exp_values / sum(exp_values)
        
        pred_error =  targets[n] - norm_values
        total_loss += np.sum(abs(pred_error))
        
        #Updating Weights and biases 
        weights += learning_rate * np.outer(inputs[n], pred_error)
        biases += learning_rate * pred_error
        
        #Calculating accuracy
        if np.argmax(norm_values) == np.argmax(targets[n]):
            correct_predictions += 1
            
    accuracy = correct_predictions / 30
    avg_loss = total_loss / 30
    
    epoch_accuracies.append(accuracy)
    epoch_losses.append(avg_loss)
    
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.2f}, Accuracy: {accuracy * 100:.0f}%")
    
    

plt.plot(range(1, epochs + 1), epoch_losses, color="black")
    
    
    
    