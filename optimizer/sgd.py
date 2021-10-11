'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-27 20:12:55
LastEditors: yp
LastEditTime: 2021-06-28 20:26:12
'''
#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay):

	print('Initialised SGD optimizer')

	return torch.optim.SGD(parameters, lr = lr, momentum = 0.9, weight_decay=weight_decay);