#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:05:46 2022

@author: gauravsingh
"""

# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays

#For model building
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MyData(Dataset):
    def __init__(self, train=True, transform=None):
        #Loading train.csv
        train_df=pd.read_csv('Img_Data.csv')
        #Loading image data and merging with train.csv
        df = pd.read_csv('Images_Class.csv')
        #Leaving only image related  columns
        feature=train_df
        #Setting labels
        label_valence=df[:,0]
        label_arousal=df[:,1]
        #Splitting the data into train and validation set
        X_train, X_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test, y_race_train,\
        = train_test_split(feature, label_valence, label_arousal, test_size=0.2)
        
        if train==True:
            self.x=X_train
            self.valence_y=y_valence_train
            self.arousal_y=y_arousal_train
        else:
            self.x=X_test
            self.valence_y=y_valence_test
            self.arousal_y=y_arousal_test            
        
        #Applying transformation
        self.transform=transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        image=np.array(self.x.iloc[idx, 0:]).astype(float).reshape(137, 236)
        label1=np.array([self.valence_y.iloc[idx]]).astype('float')
        label2=np.array([self.arousal_y.iloc[idx]]).astype('float')
        
        sample={'image': np.uint8(image), 'label_valence': label1,\
                'label_arousal': label2}
        
        #Applying transformation
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
