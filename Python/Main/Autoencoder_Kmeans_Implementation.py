"""
Description: This file implements the Autoencoder + Kmeans Unsupervised Learning method on EEG Data

"""
import sys

if sys.version[0:5] != '3.6.8':
    print("\n you are currently using python version - {},\n\n please use python 3.6.8".format(sys.version[0:5]))
    sys.exit()

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import torch
import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import os
import pandas as pd
#from StatsClassification import create_labels
import math
import random
from classification.Amigos.AmigosUtil import AmigosUtil
from sys import platform
import datetime


"""
Description: Helper Functions to separate the labels in low, mid and high valence

@params: label_dataframe, input Valence, Arousal data on the scale 0-10

@return: foramtted Emotion Label Values in three regions
"""
def create_labels(label_dataframe):
    valence_data = label_dataframe.iloc[:, 0]
    arousal_data = label_dataframe.iloc[:, 1]
    va_list = []
    aa_list = []
    for i in valence_data:
        if i <= 3.5:
            va_list.append("LV")
        if (i > 3.5) and (i <= 6.5):
            va_list.append("MV")
        if i > 6.5:
            va_list.append("HV")

    for i in arousal_data:
        if i <= 3.5:
            aa_list.append("LA")
        if (i > 3.5) and (i <= 6.5):
            aa_list.append("MA")
        if i > 6.5:
            aa_list.append("HA")

    labels_str_list = ["{}{}".format(va, aa) for va, aa in zip(va_list, aa_list)]

    labels_dict = {"LVLA": 1, "LVMA": 2, "LVHA": 3, "MVLA": 4, "MVMA": 5, "MVHA": 6, "HVLA": 7, "HVMA": 8, "HVHA": 9}
    labels_num = [labels_dict[i] for i in labels_str_list]
    return labels_num


"""
Description: Helper Functions to separate the labels in low, mid and high valence

@params: label_dataframe, input Valence, Arousal data on the scale 0-10

@return: foramtted Emotion Label Values in three regions
"""
def create_labels_ind(label_dataframe, index):
    #valence_data = label_dataframe.iloc[:, 0]
    emotion_data = label_dataframe.iloc[:, index]
    emotion_list = []
    for i in emotion_data:
        if i <= 3.5:
            emotion_list.append(1)
        if (i > 3.5) and (i <= 6.5):
            emotion_list.append(2)
        if i > 6.5:
            emotion_list.append(3)
    return emotion_list


#Access the database for retrieving the extracted features and labels
dataset_used = "amigos"  # "amigos" or "dreamer"

if "win" in platform:
	current_path = os.path.realpath(__file__).rsplit("\\", 1)[0]
	dataset_folder_path = os.path.join(current_path.rsplit("\\", 1)[0], "Data", "Extracted_features")
else:
	current_path = os.path.realpath(__file__).rsplit("/", 1)[0]
	dataset_folder_path = os.path.join(current_path.rsplit(r"/", 1)[0], "Data", "Extracted_features")

dataset_path = os.path.join(dataset_folder_path, dataset_used + ".csv")
labels_path = os.path.join(dataset_folder_path, dataset_used + "_labels.csv")

input_data = pd.read_csv(dataset_path, header=None)
input_scores_data = pd.read_csv(labels_path, header=None)

labels_values_valence = create_labels_ind(input_scores_data,0)
labels_values_arousal = create_labels_ind(input_scores_data,1)

labels_values_valence_np = np.array(labels_values_valence)
labels_values_arousal_np = np.array(labels_values_arousal)

# Create Autoencoder Architecutre using tensorflow
input_df = Input( shape = (42, ))

x = Dense(7, activation = 'relu')(input_df)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(x)

encoded = Dense(15, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(encoded)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

decoded = Dense(42, kernel_initializer='glorot_uniform')(x)


autoencoder_val = Model(input_df, decoded)

encoder_val = Model(input_df, encoded)

autoencoder_val.compile(optimizer = 'adam', loss = 'mean_squared_error')

autoencoder_arsl = Model(input_df, decoded)

encoder_arsl = Model(input_df, encoded)

autoencoder_arsl.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Load Amigos Data
amigos_data = genfromtxt(dataset_path, delimiter=',')
#print(type(amigos_data))
#print(np.shape(amigos_data))

# Scale the feature data vector
scaler = StandardScaler()
amigos_data_scaled = scaler.fit_transform(amigos_data)
#print(type(amigos_data_scaled))
#print(np.shape(amigos_data_scaled))
# m, n = np.shape(amigos_data_scaled)

#  Generate Training and Test Data for Valence
training_set_val, test_set_val, training_labels_val, test_labels_val = train_test_split(amigos_data_scaled,
                                                                       					labels_values_valence_np,
                                                                       					test_size=0.2)

#  Generate Training and Test Data for Arousal
training_set_arsl, test_set_arsl, training_labels_arsl, test_labels_arsl = train_test_split(amigos_data_scaled,
                                                                       					labels_values_arousal_np,
                                                                       					test_size=0.2)

print("Train Autoencoder for Valence and Arousal Sets ...............")
autoencoder_val.fit(training_set_val, training_set_val, batch_size= 50, epochs = 15, verbose = 1)

autoencoder_val.fit(training_set_arsl, training_set_arsl, batch_size= 50, epochs = 15, verbose = 1)

print("Autoencoder Trained Model for Valence....")
autoencoder_val.summary()
print("Autoencoder Trained Model for Arousal....")
autoencoder_arsl.summary()

print("Compress Valence and Arousal using Autoencoder....")
# Using the Autoencoder, perform Feature Compression on Valence test set
compressed_val = encoder_val.predict(test_set_val)

# Using the Autoencoder, perform Feature Compression on Arousal test set
compressed_arsl = encoder_arsl.predict(test_set_val)
#pred.shape

print("Applying Kmeans on uncompressed and compressed feature vectors....")
#Perform K Means on the initial scaled data to see the elbow curve
score_1 = []
range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(amigos_data_scaled)
    score_1.append(kmeans.inertia_)

plt.plot(score_1, 'bx-')
plt.title('Kmeans on Original Data')
plt.xlabel('Clusters')
plt.ylabel('Scores WCSS')
#plt.show()

#--k-means++ ensures not falling into random initialization trap
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
labels = kmeans.fit_predict(amigos_data_scaled)


#Auto Encoder + k-Means for valence data
score_val = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(compressed_val)
    score_val.append(kmeans.inertia_)

score_arsl = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(compressed_arsl)
    score_arsl.append(kmeans.inertia_)

kmeans = KMeans(3)
kmeans.fit(compressed_val)
kmeans_pred_labels_val = kmeans.labels_

kmeans = KMeans(3)
kmeans.fit(compressed_arsl)
kmeans_pred_labels_arsl = kmeans.labels_


print("Map compressed Kmean clusters to valence and Arousal scores to check for accuracy....")
cluster_mapping_scores_val = 1 + kmeans_pred_labels_val
cluster_mapping_scores_arsl = 1 + kmeans_pred_labels_arsl

seed = "auto_encoder"

val_accuracy, cm_val = AmigosUtil().calculate_accuracy(test_labels_val,cluster_mapping_scores_val,seed,True)
arsl_accuracy, cm_arsl = AmigosUtil().calculate_accuracy(test_labels_arsl,cluster_mapping_scores_arsl,seed,False)

print("Accuracy for Valence: ", val_accuracy)
print("Accuracy for Arousal: ", arsl_accuracy)

# Store Data in ./results
now = datetime.datetime.now()
current_time = now.strftime("%m_%d_%Y_%H_%M_%S")

i = "Autoencoder_KMeans"
save_folder = os.path.join(current_path, "results", current_time+"_Autoencoder_KMeans")

if not os.path.isdir(os.path.join(current_path, "results")):
    os.mkdir(os.path.join(current_path, "results"))
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

plt.plot(score_1, 'bx-',color = 'r',  label = 'K Means + Original Data') 
plt.plot(score_val, 'bx-',color = 'b',  label = 'K Means + Compressed Data')
#plt.plot(score_arsl, 'bx-',color = 'g',  label = 'K Means + Compressed Data for Arousal')
plt.legend()
plt.title('K Means with and without Autoencoder')
plt.xlabel('Clusters')
plt.ylabel('Scores WCSS')
#plt.show()
plt.savefig(os.path.join(save_folder, r'{}_on_{}.png'.format(i, "_elbow_curve")))


print_str = "valence"
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["low", "mid", "high"])
cm_display.plot()
plt.title('Method used - {}, tested on rand Test data'.format(i))
plt.savefig(os.path.join(save_folder, r'{}_on_{}.png'.format(i, print_str)))

conf_matrix_prnt = "Method used - {}, tested on |{}| \n accuracy percentage - {:.2f} \n true class on y axis \n {} \n\n".format(i, print_str, val_accuracy, str(cm_val))

with open(os.path.join(save_folder, "results.txt"), 'a', encoding='utf-8') as f:
    f.write(conf_matrix_prnt)

print("percentage accuracy using [{}] on {} = {:.2f}% ".format(i, print_str, val_accuracy))

print_str = "arousal"
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_arsl, display_labels=["low", "mid", "high"])
cm_display.plot()
plt.title('Method used - {}, tested on rand Test data'.format(i))
plt.savefig(os.path.join(save_folder, r'{}_on_{}.png'.format(i, print_str)))

conf_matrix_prnt = "Method used - {}, tested on |{}| \n accuracy percentage - {:.2f} \n true class on y axis \n {} \n\n".format(i, print_str, arsl_accuracy, str(cm_arsl))

with open(os.path.join(save_folder, "results.txt"), 'a', encoding='utf-8') as f:
    f.write(conf_matrix_prnt)

print("percentage accuracy using [{}] on {} = {:.2f}% ".format(i, print_str, arsl_accuracy))