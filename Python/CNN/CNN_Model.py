#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:11:10 2022

@author: gauravsingh
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import pretrainedmodels
from CNN_Run import train_dataloader,test_dataloader 



class CNN1(nn.Module):
    def __init__(self, pretrained):
        super(CNN1, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.fc1 = nn.Linear(512, 100)  #For age class
        self.fc2 = nn.Linear(512, 2)    #For gender class
        self.fc3 = nn.Linear(512, 4)    #For race class
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2= torch.sigmoid(self.fc2(x))  
        label3= self.fc3(x)
        return {'label1': label1, 'label2': label2, 'label3': label3}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Setting model and moving to device
model_CNN = CNN1(True).to(device)
#For binary output:gender
criterion_binary= nn.BCELoss()
#For multilabel output: race and age
criterion_multioutput = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion1, criterion2, optimizer, n_epochs=25):
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
            image, label1, label2, label3 = sample_batched['image'].to(device),\
                                             sample_batched['label_age'].to(device),\
                                              sample_batched['label_gender'].to(device),\
                                               sample_batched['label_race'].to(device)  
            # zero the parameter gradients
            optimizer.zero_grad()
            output=model(image)
            label1_hat=output['label1']
            label2_hat=output['label2']
            label3_hat=output['label3']         
            # calculate loss
            loss1=criterion1(label1_hat, label1.squeeze().type(torch.LongTensor))
            loss2=criterion2(label2_hat, label2.squeeze().type(torch.LongTensor))
            loss3=criterion1(label3_hat, label3.squeeze().type(torch.LongTensor))     
            loss=loss1+loss2+loss3
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
            image, label1, label2, label3 = sample_batched['image'].to(device),\
                                             sample_batched['label_age'].to(device),\
                                              sample_batched['label_gender'].to(device),\
                                               sample_batched['label_race'].to(device)  
            output = model(image)
            output=model(image)
            label1_hat=output['label1']
            label2_hat=output['label2']
            label3_hat=output['label3']               
            # calculate loss
            loss1=criterion1(label1_hat, label1.squeeze().type(torch.LongTensor))
            loss2=criterion2(label2_hat, label2.squeeze().type(torch.LongTensor))
            loss3=criterion1(label3_hat, label3.squeeze().type(torch.LongTensor))  
            loss=loss1+loss2+loss3
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