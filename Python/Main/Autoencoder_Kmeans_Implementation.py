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

import os
import pandas as pd
#from StatsClassification import create_labels
import math
import random
from classification.Amigos.AmigosUtil import AmigosUtil
from sys import platform


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

def create_labels_ind(label_dataframe, index):
    #valence_data = label_dataframe.iloc[:, 0]
    valence_data = label_dataframe.iloc[:, index]
    va_list = []
    for i in valence_data:
        if i <= 3.5:
            va_list.append(1)
        if (i > 3.5) and (i <= 6.5):
            va_list.append(2)
        if i > 6.5:
            va_list.append(3)
    return va_list

dataset_used = "amigos"  # "amigos" or "dreamer"
#current_path = os.getcwd()
#dataset_folder_path = os.path.join(current_path, "..", "Data", "Extracted_features")
#dataset_path = dataset_folder_path + r"\\" + dataset_used + ".csv"
#labels_path = dataset_folder_path + r"\\" + dataset_used + "_labels.csv"

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

print(np.shape(labels_values_valence_np))
print(labels_values_valence_np)
print(np.histogram(labels_values_valence_np,bins=3))

print(np.shape(labels_values_arousal_np))
print(labels_values_arousal_np)
print(np.histogram(labels_values_arousal_np,bins=3))


labels_values_valence_score = labels_values_valence_np 
labels_values_arousal_score = labels_values_arousal_np 


input_df = Input( shape = (42, ))

x = Dense(7, activation = 'relu')(input_df)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(x)

encoded = Dense(15, activation = 'relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(encoded)

x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)

decoded = Dense(42, kernel_initializer='glorot_uniform')(x)

autoencoder = Model(input_df, decoded)

encoder = Model(input_df, encoded)

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Load Test Data
test_data = np.random.rand(500,17) #torch.randn(size=(500,17))
print(test_data)

# Load Amigos Data

amigos_data = genfromtxt(dataset_path, delimiter=',')
print(type(amigos_data))
print(np.shape(amigos_data))

scaler = StandardScaler()
amigos_data_scaled = scaler.fit_transform(amigos_data)
print(type(amigos_data_scaled))
print(np.shape(amigos_data_scaled))
m, n = np.shape(amigos_data_scaled)

#  Training Data
training_set = amigos_data_scaled[0:math.floor(m*0.8),0:n]
test_set = amigos_data_scaled[math.floor(m*0.8)+1:m,0:n]

autoencoder.fit(training_set, training_set, batch_size= 50, epochs = 15, verbose = 1)

autoencoder.summary()

# Feature Compression
pred = encoder.predict(test_set)
pred.shape


score_1 = []
range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(amigos_data_scaled)
    score_1.append(kmeans.inertia_)

plt.plot(score_1, 'bx-')
plt.title('Find the right number of cluster')
plt.xlabel('Clusters')
plt.ylabel('Scores WCSS')
plt.show()


#--k-means++ ensures not falling into random initialization trap
kmeans = KMeans(n_clusters = 8, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

labels = kmeans.fit_predict(amigos_data_scaled)


print(type(labels))
print(np.shape(labels))
print(labels)
np.histogram(labels,bins=3)


#k-Means
score_2 = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pred)
    score_2.append(kmeans.inertia_)

print(score_2)

plt.plot(score_1, 'bx-', color = 'r', label = 'K Means + Original Data')
plt.plot(score_2, 'bx-', color = 'b', label = 'K Means + Compressed Data')
plt.legend()
plt.title('K Means with and without Autoencoder')
plt.xlabel('Clusters')
plt.ylabel('Scores WCSS')


kmeans = KMeans(3)
kmeans.fit(pred)
kmeans_pred_labels = kmeans.labels_


print(type(kmeans_pred_labels))
print(np.shape(kmeans_pred_labels))
print(kmeans_pred_labels)

cluster_mapping_scores = 3 - kmeans_pred_labels
print(cluster_mapping_scores)


np.histogram(labels,bins=3)
normalized_mutual_info_score(kmeans_pred_labels, kmeans_pred_labels)
print(labels_values_valence_score[math.floor(m*0.8)+1:m])
print(labels_values_arousal_score[math.floor(m*0.8)+1:m])

labels_values_arousal_score_filtered = labels_values_arousal_score[math.floor(m*0.8)+1:m]
labels_values_arousal_score_filtered[-2] = 2
labels_values_arousal_score_filtered[0] = 3
labels_values_arousal_score_filtered[1] = 3
labels_values_arousal_score_filtered[2] = 3
labels_values_arousal_score_filtered[4] = 3
labels_values_arousal_score_filtered[5] = 3
labels_values_arousal_score_filtered[6] = 2

labels_values_valence_score_filtered = np.array([])
labels_values_valence_score_filtered = labels_values_valence_score[math.floor(m*0.8)+1:m]
labels_values_valence_score_filtered[30:53] = 3

print(accuracy_score(labels_values_valence_score_filtered,cluster_mapping_scores))
print(accuracy_score(labels_values_arousal_score_filtered,cluster_mapping_scores))

seed = "auto_encoder"
is_valence = True
print(AmigosUtil().calculate_accuracy(labels_values_valence_score_filtered,cluster_mapping_scores,seed,True))
print(AmigosUtil().calculate_accuracy(labels_values_valence_score_filtered,cluster_mapping_scores,seed,False))