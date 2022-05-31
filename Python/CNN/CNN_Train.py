#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:26:57 2022

@author: gauravsingh
"""

from CNN_Model import model_CNN,train_model,criterion_multioutput, criterion_binary, optimizer

model_conv=train_model(model_CNN, criterion_multioutput, criterion_binary, optimizer)