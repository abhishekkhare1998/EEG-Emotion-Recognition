#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:14:59 2022

@author: gauravsingh
"""

import torch
from CNN_Data_Process import MyData
from torch.utils.data import DataLoader
import CNN_Model
import numpy as np
from torch import nn, optim
from torch.nn import functional as F


class CNN1(nn.Module):
    def __init__(self, pretrained):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 16)
        self.conv2 = nn.Conv2d(8, 8, 8)
        self.fc1 = nn.Linear(3872, 256)
        self.pool1 = nn.MaxPool2d(8)
        self.pool2 = nn.MaxPool2d(4)
        self.fc21 = nn.Linear(256, 3)    #For valence class
        self.fc22 = nn.Linear(256, 3)    #For arousal class
        
    def forward(self, x):
        
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        label1 = self.fc21(x)
        label1 = F.log_softmax(label1, dim=1)

        label2 = self.fc22(x)
        label2 = F.log_softmax(label2, dim=1)
        return {'label1': label1, 'label2': label2}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Setting model and moving to device
model_CNN = CNN1(True).to(device)
#For multilabel output: race and age
criterion_multioutput = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion1, optimizer, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label1, label2 = sample_batched['image'].to(device),\
                                             sample_batched['label_valence'].to(device),\
                                              sample_batched['label_arousal'].to(device) 
            # zero the parameter gradients
            optimizer.zero_grad()
            output=model(image)
            label1_hat=output['label1']
            label2_hat=output['label2']        
            # calculate loss
            loss1=criterion1(label1_hat, label1.squeeze().type(torch.LongTensor))
            loss2=criterion1(label2_hat, label2.squeeze().type(torch.LongTensor))    
            loss=loss1+loss2
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        # validate the model #
        model.eval()
        for batch_idx, sample_batched in enumerate(test_dataloader):
            image, label1, label2 = sample_batched['image'].to(device),\
                                             sample_batched['label_valence'].to(device),\
                                              sample_batched['label_arousal'].to(device)  
            output = model(image)
            output=model(image)
            label1_hat=output['label1']
            label2_hat=output['label2']              
            # calculate loss
            loss1=criterion1(label1_hat, label1.squeeze().type(torch.LongTensor))
            loss2=criterion1(label2_hat, label2.squeeze().type(torch.LongTensor))
            loss=loss1+loss2
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
    # return trained model
    return model

transformed_train_data = MyData(train=True, transform=None)
transformed_test_data = MyData(train=False, transform=None)
train_dataloader = DataLoader(transformed_train_data, batch_size=50, shuffle=True, num_workers=4)
test_dataloader = DataLoader(transformed_test_data, batch_size=50, shuffle=True, num_workers=4)

model_conv=train_model(model_CNN, criterion_multioutput, optimizer)