#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:09:52 2023

@author: noederijck
"""

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


plt.style.use('seaborn-dark-palette')
palette = sns.color_palette("Set2")

#PARAMETERS THAT CAN BE MODIFIED
num_letter = 9
number_of_pairs = 8000
num_hid_neuron = (num_letter*26)/8
num_epochs = 200000
num_subsets = 5
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
    subset_pred_dic = {}
    for num in range(len(subset_train)):
        subset_pred = model.predict(subset_train[num])
        subset_error = np.mean(np.abs(subset_targets[num] - subset_pred))
        subset_PE_dict[num] = subset_error
        subset_pred_dic[num] = subset_pred
    choice, index = strategy(PE_dic = subset_PE_dict, strategy = strat)
    subset_choice = np.random.choice(len(subset_en_train), size=1, p=choice)
    if subset_choice == index:
        chosen_start = 1
    else:
        chosen_start = 0
    return subset_choice[0], chosen_start, subset_pred_dic, subset_PE_dict


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
        print(" ")
        print(average_PE)
        print(PE_dic.values())
        print(" ")
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
    if PE_stratergy == "rand":
        color_map = {1: 'black', 0: 'black'}        
    try:
        if len(colors) == 5:
            colors_sub = colors_sub = [(1.0, 0.8509803921568627, 0.1843137254901961) ,(0.9058823529411765, 0.5411764705882353, 0.7647058823529411),(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),(1.0, 0.8509803921568627, 0.1843137254901961),(1.0, 0.8509803921568627, 0.1843137254901961)]
        else:
            colors_sub = [random.choice(palette) for _ in range(len(group_y_axis))]
        for subset in range(len(group_y_axis)):
            color_map = {1: 'green', 0: colors_sub[subset]}
            #color_map = {1: colors_sub[subset], 0: colors_sub[subset]}
            data_points = np.arange(len(group_y_axis[subset]))
            for i in range(len(group_y_axis[subset]) - 1):
                start = (data_points[i], group_y_axis[subset][i])
                stop = (data_points[i + 1], group_y_axis[subset][i + 1])
                color = color_map[colors[subset][i]]
                plt.plot([start[0], stop[0]], [start[1], stop[1]], color=color)
    except:
        color_map = {1: 'green', 0: "green"}
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
list_average_PE = []


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
subset_en_train = np.array_split(en_train, num_subsets)
subset_fr_labels = np.array_split(fr_labels, num_subsets)


acc_subsets = {}

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

#How many models trained until 80% do you want
# "avg" for average PE
# "min" for minimum PE
# "max" for maximum PE
# "rand" for random PE
for model_index in range(1):
    st = time.time()
    if model_index == 1:
        PE_stratergy = "max"
    elif model_index == 0:
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
        while training_accuracy < 0.5:
            history = model.fit(x=en_train, y=fr_labels, batch_size=30, shuffle=True, verbose=0)
            prediction = model.predict(en_train)
            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0           
            training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
            print((f"Epoch: {epoch} Accuracy: {training_accuracy:.3f}"))
            epoch += 1
        model.save("letter_semi_trained_random_in_out.keras")
    
    
    last_accuracy = 0
    counter = 0
    epoch = 0
    while epoch < num_epochs and training_accuracy < 0.95:
        choice, chosen_strat, subset_pred_dic, subset_PE_dict = choice_subset(subset_train = subset_en_train, subset_targets = subset_fr_labels, strat = PE_stratergy)
        for i in range(len(subset_pred_dic)):
            subset_pred_dic[i][subset_pred_dic[i]>0.5] = 1
            subset_pred_dic[i][subset_pred_dic[i]<=0.5] = 0
        print((f"Epoch: {epoch} Accuracy: {training_accuracy:.2f}"))
        for i in range(steps_to_switch_strat):
            if training_accuracy < 0.95:
                    history = model.fit(x=subset_en_train[choice], y=subset_fr_labels[choice], batch_size=30, shuffle=True, verbose=0)
                    choice_log.append(choice)
                    prediction = model.predict(en_train)
                    prediction[prediction>0.5] = 1
                    prediction[prediction<=0.5] = 0
                    list_average_PE.append(np.mean(list(subset_PE_dict.values())))
                    for i in range(len(subset_pred_dic)):
                        subset_accuracy = np.all(subset_pred_dic[i] == subset_fr_labels[i], axis=1).sum()/subset_pred_dic[i].shape[0]
                        acc_subsets[i].append(subset_accuracy)
                    PE = np.mean(abs(prediction - fr_labels))
                    PEs.append(PE)
                    training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
                    accuracies.append(training_accuracy)
                    losses.append(history.history["loss"][0])
                    strat_log.append(chosen_strat)
                    epoch +=1
        if epoch % 2000 == 0:
            print("Training accuracy =",round(training_accuracy, 2), "Last accuracy =",round(last_accuracy, 2))
            if round(training_accuracy, 2) <= round(last_accuracy, 2):
                #counter += 1
                print(f"{counter} Warning !")
                if counter == 4:
                    num_epochs = epoch  + 50
            else:
                last_accuracy = training_accuracy
    
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
        
    plt.plot(PEs)
    plt.title('PEs')
    plt.xlabel('Epochs')
    plt.ylabel('PEs')
    plt.show()
    accuracies = []
    losses = []
    strat_log = []
    choice_log = []
    PEs = []
    for sub in range(len(subset_fr_labels)):
        acc_subsets[sub] = []
    
