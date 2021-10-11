'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-15 14:19:57
LastEditors: yp
LastEditTime: 2021-06-15 14:20:18
'''
'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-13 15:08:59
LastEditors: yp
LastEditTime: 2021-06-15 11:16:38
'''

from numpy.lib.histograms import _get_outer_edges
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy
from sklearn.metrics import accuracy_score

class LossFunction(nn.Module):
    def __init__(self, nOut, age_Classes = 1, gender_Classes = 2,margin=0.3, scale=15,mode = "train", **kwargs):
        self.mode = mode
        layers = [1024,512]
        super(LossFunction, self).__init__()
        self.fc_a_1 = nn.Linear(nOut,layers[0])
        self.fc_a_2 = nn.Linear(layers[0],layers[1])
        self.fc_a_3 = nn.Linear(layers[1],1)
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        self.batch_norm3 = nn.BatchNorm1d(layers[0])
        self.batch_norm4 = nn.BatchNorm1d(layers[1])
        self.fc_g_1 = nn.Linear(nOut,layers[0])
        self.fc_g_2 = nn.Linear(layers[0],layers[1])
        self.fc_g_3 = nn.Linear(layers[1],gender_Classes)
        self.ce = torch.nn.CrossEntropyLoss()
        


    def forward(self, x):


        a_1 = F.relu(self.batch_norm1(self.fc_a_1(x)))
        a_2 = F.relu(self.batch_norm2(self.fc_a_2(a_1)))

        g_1 = F.relu(self.batch_norm3(self.fc_g_1(x)))
        g_2 = F.relu(self.batch_norm4(self.fc_g_2(g_1)))

        a_out = self.fc_a_3(a_2)
        g_out = self.fc_g_3(g_2)
        # if self.mode == "train" or self.mode == "dev":
        #     age_pre    ,  age_loss       =  self.get_loss(a_out, age_label)
        #     gender_pre ,  gender_loss   =  self.get_loss(g_out, gender_label)
        #     if weight >= 1:
        #         loss = age_loss * weight + gender_loss
        #     else:
        #         loss = age_loss * weight + (1 - weight) * gender_loss
        #     return loss,age_loss,gender_loss,age_pre,gender_pre
        # else:
        #     a_out,g_out = get_pre(a_out,g_out)
        a_out = a_out.squeeze(-1)
        return a_out, g_out

    def get_loss(self,x,label):

        loss = self.ce(x, label)
        preds = []
        labels = []
        precinos = numpy.argmax(x.detach().cpu().numpy(), axis=1)
        for labet in label.detach().cpu().numpy():
            labels.append(labet)
        for pre in precinos:
            preds.append(pre)
        # print("**********   label   *************")
        # print(labels)
        # print("**********   preds   *************")
        # print(preds)
        acc = accuracy_score(labels,preds)
        # print(acc)
        return acc,loss

    def get_pre(a_out,g_out):
        a_preds,g_preds = [],[]
        precinos = numpy.argmax(a_out.detach().cpu().numpy(), axis=1)
        for pre in precinos:
            a_preds.append(pre)
        precinos = numpy.argmax(g_out.detach().cpu().numpy(), axis=1)
        for pre in precinos:
            g_preds.append(pre)
        return a_preds,g_preds