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
st = time.time()
#chooses the subset of words depending on the strategy 
def strategy(PE_dic, strategy):
    #Creats an array of length equal to number of subsets
    P_distribution = np.full(len(PE_dic), (1/(len(PE_dic)*((len(PE_dic)-1)/2))))
    #P_distribution = np.full(len(PE_dic), 0.4*(len(PE_dic)-1))
    if strategy == "max":
        #calculates which subset has the highest PE
        choice = max(zip(PE_dic.values(), PE_dic.keys()))[1] 
        P_distribution[choice] = 1 - (len(PE_dic) - 1) / (len(PE_dic) * 2)
    elif strategy == "min":
        #calculates which subset has the lowest PE
        
        choice = min(zip(PE_dic.values(), PE_dic.keys()))[1]
        P_distribution[choice] = 1 - (len(PE_dic) - 1) / (len(PE_dic) * 2)
    elif strategy == "avg":
        #calculates which subset has the most average PE
        average_PE = np.mean(list(PE_dic.values()))
        choice = min(PE_dic.items(), key=lambda x: abs(average_PE - x[1]))[0]
        #Sets the probability of the subset chosen by the strategy equal the inverse of the sum of all the other probabilities 
        P_distribution[choice] = 1 - (len(PE_dic) - 1) / (len(PE_dic) * 2)
    elif strategy == "rand":
        #calculates which subset has the most average PE
        P_distribution = np.full(len(PE_dic), (1/(len(PE_dic))))
    return P_distribution
    

#calculates PE for each subset of words and returns the index of the chosen subset (60% chance it chooses it's prefered strategy if there are 5 subsets)
def choice_subset(subset_train, subset_targets, strat):
    subset_PE_dict = {}
    for num in range(len(subset_train)):
        subset_pred = model.predict(subset_train[num])
        subset_error = np.mean(np.abs(subset_targets[num] - subset_pred))
        subset_PE_dict[num] = subset_error
    choice = strategy(PE_dic = subset_PE_dict, strategy = strat)
    subset_choice = np.random.choice(len(subset_en_train), size=1, p=choice)
    return subset_choice[0]


# Plots the accuracy/loss but changes the title based on what stategy was used
def plots(y_axis):
    global PE_stratergy
    plt.plot(y_axis, color="black")
    if PE_stratergy == "avg":
        plt.title('Average PE strategy')
        
    if PE_stratergy == "min":
        plt.title('Minimum PE strategy')
        
    if PE_stratergy == "max":
        plt.title('Maximum PE strategy')
        
    if PE_stratergy == "rand":
        plt.title('Random PE strategy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    
    
random.seed(1999)
words_list = []
epoch = 0
fr_labels = []
en_train = []
num_letter = 6
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



#Import fr/en word pairs
with open("/Users/noederijck/Desktop/word_lists/fr_en_long_word_list.txt", "r") as file:
    for line in file:
        word_pair = line.strip().split(",")
        words_list.append(word_pair)
        
words_list_pd = pd.DataFrame(words_list)
#selects the words shorter than the designated number of letters (num_letter)
words_list_pd = words_list_pd.loc[words_list_pd[0].str.len() <= num_letter, :]
words_list_pd = words_list_pd.loc[words_list_pd[1].str.len() <= num_letter, :]

words_list_pd.index = range(words_list_pd.shape[0])

fr_labels = np.zeros([words_list_pd.shape[0], 26*num_letter])
en_train = np.zeros([words_list_pd.shape[0], 26*num_letter])

#creats multi-hot encoded inputs of length 26* number of letter
for row in range(words_list_pd.shape[0]):
    en_word = words_list_pd.iloc[row, 0]
    fr_word = words_list_pd.iloc[row, 1]

    for i in range(4):
        try:
            letter = en_word[i]
            active = i*26 + alphabet[letter]
            en_train[row, active] = 1
        except:
            pass
    
        try:
            letter = fr_word[i]
            active = i*26 + alphabet[letter]
            fr_labels[row, active] = 1
        except:
            pass


# Turns list of list into matrix of tensors
fr_labels_array = fr_labels.copy()
en_train = tf.stack(np.array(en_train))
fr_labels = tf.stack(np.array(fr_labels))

#Divides training pairs into subsets
subset_en_train = np.array_split(en_train, 5)
subset_fr_labels = np.array_split(fr_labels, 5)


#Model Parameters
optimizer = Adam(learning_rate=0.001)


#Model Structure
model = Sequential()
model.add(InputLayer(input_shape=(num_letter*26,)))
model.add(Dense(units=num_letter*26, activation='relu'))
model.add(Dense(units=num_letter*26, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

accuracies = []
losses = []

training_accuracy = 0

#Loads a model trained at 30% if available
try:
    model = tf.keras.models.load_model("letter_semi_trained.keras")
except:
    pass

#If 30% trained model not available, trains a new model until 30% accuracy
while training_accuracy < 0.3:
    history = model.fit(x=en_train, validation_split=0.1, y=fr_labels, batch_size=30, shuffle=True, verbose=0)
    prediction = model.predict(en_train)
    prediction[prediction>0.5] = 1
    prediction[prediction<=0.5] = 0
    
    training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
    accuracies.append(training_accuracy)
    print((f"Epoch: {epoch} Accuracy: {training_accuracy:.3f}"))
    epoch += 1

#Determines the strategy the agent will choose
# "avg" for average PE
# "min" for minimum PE
# "max" for maximum PE
# "rand" for random PE
PE_stratergy = "min"


epoch = 0
while training_accuracy < 0.80 and epoch < 20000:
    choice = choice_subset(subset_train = subset_en_train, subset_targets = subset_fr_labels, strat = PE_stratergy)
    for i in range(50):
        history = model.fit(x=subset_en_train[choice], validation_split=0.1, y=subset_fr_labels[choice], batch_size=30, shuffle=True, verbose=0)
        
        prediction = model.predict(en_train)
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        
        training_accuracy = np.all(prediction == fr_labels_array, axis=1).sum()/prediction.shape[0]
        accuracies.append(training_accuracy)
        losses.append(history.history["loss"])
        print((f"Epoch: {epoch} Accuracy: {training_accuracy:.2f}"))
        epoch +=1


#with open("/Users/noederijck/Desktop/word_lists/list_epoch_average.txt", "w") as file:
#    file.write(f"{epoch}\n")
        

plots(accuracies)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
