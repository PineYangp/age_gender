'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-28 09:46:45
LastEditors: yp
LastEditTime: 2021-06-28 20:24:55
'''
#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval,  lr_decay):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step
