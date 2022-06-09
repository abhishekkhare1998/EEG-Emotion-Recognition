#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:05:46 2022

@author: gauravsingh
"""

# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import mat73 as mt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MyData(Dataset):
    
    # Setting up the dataset
    def __init__(self, train=True, transform=None):
        
        #Loading Img_Data.csv
        train_df=mt.loadmat("all_img_amigos.mat")
        feature = np.array(train_df['Img_Cell'], dtype='uint8')
        feature = np.uint8(np.reshape(feature,[feature.shape[0],3,128,128]))
        #Loading Images labels from Images_Classes.csv
        df = mt.loadmat('Images_Class.mat')
        #Leaving only image related  columns
        label_int = np.array(df['k'], dtype='uint8')
        
        #Setting labels
        label_valence=label_int[:,0]
        label_arousal=label_int[:,1]
        
        #Splitting the data into train and validation set
        X_train, X_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test\
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
    
    #Getting the item from the dataset
    def __getitem__(self, idx):
        image=torch.from_numpy(self.x[idx]).type(torch.ByteTensor)
        label1=self.valence_y[idx]
        label2=self.arousal_y[idx]
        
        sample={'image': image, 'label_valence': label1,\
                'label_arousal': label2}
        
        #Applying transformation
        if self.transform:
            sample=self.transform(sample)
            
        return sample
    
