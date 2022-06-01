#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:05:46 2022

@author: gauravsingh
"""

# General libraries
import pandas as pd  #For working with dataframes
import numpy as np   #For working with image arrays
import cv2          #For transforming image

#For model building
import torch
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp


class MyData(Dataset):
    def __init__(self, train=True, transform=None):
        #Loading train.csv
        train_df=pd.read_csv('train.csv')
        #Loading image data and merging with train.csv
        df=pd.merge(pd.read_parquet('image_train_data.parquet'),\
        train_df, on='image_id').drop(['image_id'], axis=1)
        #Leaving only image related  columns
        feature=df.drop(['valence', 'arousal'], axis=1)
        #Setting labels
        label_valence=df['valence']
        label_arousal=df['arousal']
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
    
class crop(object):
    def __init__(self, resize_size):
        self.resize_size = resize_size
    def __call__(self, sample):
        image, label1, label2 = sample['image'],\
        sample['label_valence'], sample['label_arousal']
        _, thresh=cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        idx=0
        ls_xmin=[]
        ls_ymin=[]
        ls_xmax=[]
        ls_ymax=[]
        
        for cnt in contours:
            idx+=1
            x,y,w,h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)
        roi = image[ymin:ymax,xmin:xmax]    
        resized_image = cv2.resize(roi, (self.resize_size, self.resize_size),\
                                                             interpolation=cv2.INTER_AREA)
        sample={'image': resized_image, 'label_valence': label1, 'label_arousal': label2}
        return sample

class rotate_image(object):
    def __call__(self, sample):
        image, label1, label2 = sample['image'],\
        sample['label_valence'], sample['label_arousal']
        min_scale = 0.8
        max_scale = 1.2
        sx = np.random.uniform(min_scale, max_scale)
        sy = np.random.uniform(min_scale, max_scale)
        # --- rotation ---
        max_rot_angle = 7
        rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.
        # --- shear ---
        max_shear_angle = 10
        shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.
        # --- translation ---
        max_translation = 4
        tx = np.random.randint(-max_translation, max_translation)
        ty = np.random.randint(-max_translation, max_translation)
        tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
        transformed_image = warp(image, tform)
        assert transformed_image.ndim == 2
        sample={'image': transformed_image, 'label_valence': label1, 'label_arousal': label2}
        return sample

class RGB_ToTensor(object):
    def __call__(self, sample):
        image, label1, label2 = sample['image'],\
        sample['label_valence'], sample['label_arousal']
        
        image=torch.from_numpy(image).unsqueeze_(0).repeat(3, 1, 1)
        label1=torch.from_numpy(label1)
        label2=torch.from_numpy(label2)
        
        return {'image': image,
                'label_valence': label1,
                'label_arousal': label2}

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def __call__(self, sample):
        image, label1, label2 = sample['image'],\
        sample['label_valence'], sample['label_arousal']
        
        return {'image': image,
                'label_valence': label1,
                'label_arousal': label2}

    