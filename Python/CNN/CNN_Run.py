#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:14:59 2022

@author: gauravsingh
"""

import torch
from CNN_Data_Process import MyData,crop,rotate_image,RGB_ToTensor,Normalization,DataLoader
from torchvision import transforms


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
transformed_train_data = MyData(train=True, transform=transforms.Compose([crop(256),\
                                            rotate_image(), RGB_ToTensor(),
                                            Normalization(cnn_normalization_mean,\
                                            cnn_normalization_std)]))
transformed_test_data = MyData(train=False, transform=transforms.Compose([crop(256),\
                                            RGB_ToTensor(),
                                            Normalization(cnn_normalization_mean,\
                                            cnn_normalization_std)]))
train_dataloader = DataLoader(transformed_train_data, batch_size=50, shuffle=True, num_workers=4)
test_dataloader = DataLoader(transformed_test_data, batch_size=50, shuffle=True, num_workers=4)