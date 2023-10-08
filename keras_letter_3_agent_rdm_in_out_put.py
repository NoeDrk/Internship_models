#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:48:46 2023

@author: noederijck
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import time
import seaborn as sns


palette = sns.color_palette("Set2")

#PARAMETERS THAT CAN BE MODIFIED
num_letter = 9
number_of_pairs = 8000
num_hid_neuron = (num_letter*26)/8
num_epochs = 60000
steps_to_switch_strat = 50
#Probability choice strategy
p_choice = 1
#Learning rate
a = 0.001
#Determines the strategy the agent will choose
# "avg" for average PE
# "min" for minimum PE
# "max" for maximum PE
# "rand" for random PE
PE_stratergy = "avg"



#calculates PE for each subset of words and returns the index of the chosen subset (60% chance it chooses it's prefered strategy)
def choice_subset(subset_train, subset_targets, strat):
    subset_PE_dict = {}
    for num in range(len(subset_train)):
        subset_pred = model.predict(subset_train[num])
        subset_error = np.mean(np.abs(subset_targets[num] - subset_pred))
        subset_PE_dict[num] = subset_error
    choice, index = strategy(PE_dic = subset_PE_dict, strategy = strat)
    subset_choice = np.random.choice(len(subset_en_train), size=1, p=choice)
    if subset_choice == index:
        chosen_start = 1
    else:
        chosen_start = 0
    return subset_choice[0], chosen_start


def strategy(PE_dic, strategy):
    #Creats an array of length equal to number of subsets
    P_distribution = np.full(len(PE_dic), (1-p_choice)/(len(PE_dic)-1))
    #P_distribution = np.full(len(PE_dic), 0.4*(len(PE_dic)-1))
    if strategy == "max":
        #calculates which subset has the highest PE
        choice = max(zip(PE_dic.values(), PE_dic.keys()))[1] 
        #P_distribution[choice] = 1 - (len(PE_dic) - 1) / (len(PE_dic) * 2)
        P_distribution[choice] = p_choice
    elif strategy == "min":
        #calculates which subset has the lowest PE
        choice = min(zip(PE_dic.values(), PE_dic.keys()))[1]
        P_distribution[choice] = p_choice
    elif strategy == "avg":
        #calculates which subset has the most average PE
        average_PE = np.mean(list(PE_dic.values()))
        choice = min(PE_dic.items(), key=lambda x: abs(average_PE - x[1]))[0]
        #Sets the probability of the subset chosen by the strategy equal the inverse of the sum of all the other probabilities 
        P_distribution[choice] = p_choice
    elif strategy == "rand":
        #calculates which subset has the most average PE
        P_distribution = np.full(len(PE_dic), (1/(len(PE_dic))))
        choice = 99
    return P_distribution, choice
    

        
# Plots the accuracy/loss but changes the title based on what stategy was used
def plots(group_y_axis, g_type, colors):
    plt.figure(figsize=(12.8, 9.6))
    global PE_strategy
  #Switches the color of the line when the strategy that was chosen is used
    if PE_stratergy == "rand":
        color_map = {1: 'black', 0: 'black'}        
      #This paragraph works if the group_y_axis is a dictionary with subsets as keys and lists of accuracies as values example group_y_axis = {0:[n, n+1, n+T],1:[n, n+1, n+T],2:[n, n+1, n+T]}
    try:
        if len(colors) == 5:
            colors_sub = [(1.0, 0.8509803921568627, 0.1843137254901961) ,(0.9058823529411765, 0.5411764705882353, 0.7647058823529411),(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),(1.0, 0.8509803921568627, 0.1843137254901961),(1.0, 0.8509803921568627, 0.1843137254901961)]
        else:
            colors_sub = [random.choice(palette) for _ in range(len(group_y_axis))]
        for subset in range(len(group_y_axis)):
            color_map = {1: 'green', 0: colors_sub[subset]}
            data_points = np.arange(len(group_y_axis[subset]))
            for i in range(len(group_y_axis[subset]) - 1):
                start = (data_points[i], group_y_axis[subset][i])
                stop = (data_points[i + 1], group_y_axis[subset][i + 1])
                color = color_map[colors[subset][i]]
                plt.plot([start[0], stop[0]], [start[1], stop[1]], color=color)
         #This paragraph works if the group_y_axis is a lists of values
    except:
        color_map = {1: 'green', 0: "red"}
        data_points = np.arange(len(group_y_axis))
        for i in range(len(group_y_axis) - 1):
            start = (data_points[i], group_y_axis[i])
            stop = (data_points[i + 1], group_y_axis[i + 1])
            color = color_map[colors[i]]
            plt.plot([start[0], stop[0]], [start[1], stop[1]], color=color)
        
    if PE_stratergy == "avg":
        plt.title('Average PE strategy')
    if PE_stratergy == "min":
        plt.title('Minimum PE strategy')
    if PE_stratergy == "max":
        plt.title('Maximum PE strategy')
    if PE_stratergy == "rand":
        plt.title('Random PE strategy')

    plt.xlabel('Epoch')
    plt.ylabel(g_type)
    plt.show()
    
    
    
random.seed(1999)
words_list = []
epoch = 0
fr_labels = []
en_train = []
acc_subsets = {}

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



#CODE TO MAKE THE RANDOM INPUT/OUTPUTS
#Generates arrays of len(26 * the maximum number of letters in a word)
all_inputs = np.zeros(shape=(number_of_pairs, 26*num_letter))
all_targets = np.zeros(shape=(number_of_pairs, 26*num_letter))

#Generates random input/output words with varying length
for i in range(len(all_targets)):
    #Creates a matrix (26,9) one hot encoded arrays of length 26 which are then shuffled
    in_word = np.eye(26)[np.random.choice(26, 9)]
    targ_word = np.eye(26)[np.random.choice(26, 9)]
    #Turns the matrix into a single array of len(26*9)=234
    in_word_flat = in_word.flatten()
    targ_word_flat = targ_word.flatten()
    #Generates a random number taken from a random distribution with SD=1 and Mean=5 and then rounds it up to the nearest interger
    rand_num_let_input = round(np.random.normal(loc=5,scale=1, size=1)[0])
    rand_num_let_targ = round(np.random.normal(loc=5,scale=1, size=1)[0])
    #Determines the length of the word by assigning the last X number of letters to be = 0
    in_word_flat[rand_num_let_input*26:] = 0
    targ_word_flat[rand_num_let_targ*26:] = 0
    all_inputs[i] = in_word_flat
    all_targets[i] = targ_word_flat


en_train = all_inputs
fr_labels = all_targets

#Divides training pairs into subsets
fr_labels_array = fr_labels.copy()
subset_en_train = np.array_split(en_train, 5)
subset_fr_labels = np.array_split(fr_labels, 5)

for sub in range(len(subset_fr_labels)):
    acc_subsets[sub] = []

#Model Parameters
optimizer = Adam(learning_rate=a)

#Model Structure
model = Sequential()
model.add(InputLayer(input_shape=(num_letter*26,)))
model.add(Dense(units=num_hid_neuron, activation='relu'))
model.add(Dense(units=num_letter*26, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Which statergies do you want to test ?
for model_index in range(3):
    st = time.time()
    if model_index == 0:
        PE_stratergy = "max"
    elif model_index == 1:
        PE_stratergy = "avg"
    #elif model_index == 2:
    #    PE_stratergy = "min"
    elif model_index == 2:
        PE_stratergy = "rand"
    accuracies = []
    losses = []
    strat_log = []
    choice_log = []
    PEs = []
    training_accuracy = 0
    #Loads a model trained at 30% if available
    try:
        print("Model loaded")
        print("Strategy used:", PE_stratergy)
        model = tf.keras.models.load_model("letter_semi_trained_random_in_out.keras")
    except:
        #If 30% trained model not available, trains a new model until 30% accuracy
        print("Model wasn't found")
        while training_accuracy < 0.3:
            history = model.fit(x=en_train, validation_split=0.1, y=fr_labels, batch_size=30, shuffle=True, verbose=0)
            prediction = model.predict(en_train)
            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0           
            training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
            accuracies.append(training_accuracy)
            losses.append(history.history["loss"][0])
            print((f"Epoch: {epoch} Accuracy: {training_accuracy:.3f}"))
            epoch += 1
        model.save("letter_semi_trained_random_in_out.keras")
    
    last_accuracy = 0
    counter = 0
    epoch = 0
    while epoch < num_epochs and training_accuracy < 0.95:
        choice, chosen_strat = choice_subset(subset_train = subset_en_train, subset_targets = subset_fr_labels, strat = PE_stratergy)
        print((f"Epoch: {epoch} Accuracy: {training_accuracy:.2f}"))
        for i in range(steps_to_switch_strat):
            if training_accuracy < 0.95:
                    history = model.fit(x=subset_en_train[choice], y=subset_fr_labels[choice], batch_size=30, shuffle=True, verbose=0)
                    choice_log.append(choice)
                    prediction = model.predict(en_train)
                    prediction[prediction>0.5] = 1
                    prediction[prediction<=0.5] = 0
              #Calculates the PE for each subset and adds it to a dic
                    for i in range(len(subset_en_train)):
                        pred_sub = model.predict(subset_en_train[i])
                        pred_sub[pred_sub>0.5] = 1
                        pred_sub[pred_sub<=0.5] = 0
                        subset_accuracy = np.all(pred_sub == subset_fr_labels[i], axis=1).sum()/prediction.shape[0]
                        acc_subsets[i].append(subset_accuracy)
                    PE = np.mean(abs(prediction - fr_labels))
                    PEs.append(PE)
                    training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
                    accuracies.append(training_accuracy)
                    losses.append(history.history["loss"][0])
                    strat_log.append(chosen_strat)
                    epoch +=1
              #Ends training if the accuracy has stagnated for more than 2000 epochs
        if epoch % 500 == 0:
            if training_accuracy <= last_accuracy:
                counter += 1
                print(f"{counter} Warning !")
                if counter == 4:
                    num_epochs = epoch  + 50
            else:
                temp_accuracy = training_accuracy
    
    print((f"Epoch: {epoch} Accuracy: {training_accuracy:.2f}"))
    #with open("/Users/noederijck/Desktop/word_lists/list_epoch_average.txt", "w") as file:
    #    file.write(f"{epoch}\n")
    temp_list = np.zeros(shape=(len(acc_subsets.values()), len(acc_subsets[0])))
    for l in range(len(temp_list)):
        for i in range(len(choice_log)):
            if choice_log[i] == l:
                temp_list[l][i] = 1
            else:
                temp_list[l][i] = 0
        
    plots(acc_subsets, "Accuracy", temp_list)
    plots(accuracies, "Accuracy", strat_log)
    plots(losses, "Loss", strat_log)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    with open("/Users/noederijck/Desktop/word_lists/epochs_strat.csv","a") as file:
        file.write(f"{model_index}, {len(accuracies)}, {PE_stratergy}, {elapsed_time:.2f}\n")
    #plots the abs(PE) on the X axis and the accuracy on the Y axis
    plt.scatter(PEs,accuracies, marker='o', c='black')
    plt.title('Accuracy vs. PE')
    plt.xlabel('PE')
    plt.ylabel('Accuracy')
    plt.show()
    
