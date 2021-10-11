'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-13 15:40:44
LastEditors: yp
LastEditTime: 2021-06-29 20:44:45
'''
#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
# from tuneThreshold import tuneThresholdfromScore
from Age_dataloader import loadWAV
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
import numpy as np

class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, weight,scheduler, trainfunc,use_gpu, **kwargs):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu) 
        use_cuda = torch.cuda.is_available()
        self.weight = weight
        print(use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        super(SpeakerNet, self).__init__()
        self.scheduler = scheduler
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.lr_decay = 0.2
        self.optimizer = optimizer
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).to(self.device)
        self.__S__ = nn.DataParallel(self.__S__)

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).to(self.device)

        self.Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = self.Optimizer(self.parameters(),self.lr,self.weight_decay)
        # Optimizer(filter(lambda p: p.requires_grad, self.parameters()), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, test_interval = 1,  lr_decay = self.lr_decay)
        self.loss_fun = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        assert self.lr_step in ['epoch', 'iteration']

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train()
        
        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0
        top1    = 0     # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:
            data_time = time.time() -tstart

            data = data.transpose(0,1)

            self.zero_grad()

            feat = []
            for inp in data:
                outp      = self.__S__.forward(inp.to(self.device))
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()

            label   = torch.LongTensor(data_label).to(self.device)

            nloss, prec1 = self.__L__.forward(feat,label)

            loss    += nloss.detach().cpu()
            top1    += prec1
            counter += 1
            index   += stepsize

            nloss.backward()
            self.__optimizer__.step()

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index))
            sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz Time %.3f| %.3f"%(loss/counter, top1/counter, stepsize/telapsed , data_time,telapsed))
            sys.stdout.flush()

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n")
        
        return (loss/counter, top1/counter)

    def train_age_network(self, loader):

        self.train()

        # Optimizer = importlib.import_module('optimizer.'+self.optimizer).__getattribute__('Optimizer')
        # __optimizer__ = Optimizer(filter(lambda p: p.requires_grad, self.parameters()), self.lr,self.weight_decay)
        
        # Scheduler = importlib.import_module('scheduler.'+self.scheduler).__getattribute__('Scheduler')
        # __scheduler__, lr_step = Scheduler(__optimizer__, test_interval = 1,  lr_decay = self.lr_decay)

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0
        age_top1    = 0     # EER or accuracy
        gender_top1 = 0
        age_rmse_top_1 = 0


        tstart = time.time()
        train_loss_list = list()
        for data, age_label,weight,gender_label in loader:
            train_loss_list = []
            full_gender_preds = []
            full_age_preds = []
            full_gender_gts = []
            full_age_gts = []

            data_time = time.time() -tstart

            data = data.transpose(0,1)

            self.zero_grad()

            feat = []
            for inp in data:
                outp      = self.__S__.forward(inp.to(self.device))
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()

            age_label   = age_label.to(self.device)
            # age_label = age_label.float()
            gender_label = torch.LongTensor(gender_label).to(self.device)

            a_out,g_out = self.__L__.forward(feat)
            
            # age_loss = F.mse_loss(a_out,age_label)
            # age_loss = self.mse(a_out,age_label)
            weight = weight.to(self.device)
            age_loss = weighted_focal_mse_loss(a_out,age_label,weights = weight)
            # age_loss = self.mse(a_out,age_label)

            gender_loss = self.loss_fun(g_out,gender_label)
            # weight_age = age_loss * age_weight
            # loss = weight_age + gender_loss
            nloss =  self.weight * age_loss + gender_loss
            nloss.backward()

            train_loss_list.append(nloss.item())

            # age_predictions = numpy.argmax(a_out.detach().cpu().numpy(), axis=1)
            
            age_predictions = 100 * a_out.detach().cpu().numpy()
            age_label *= 100
            gender_predictions = numpy.argmax(g_out.detach().cpu().numpy(), axis=1)
            for age_pred in age_predictions:
                full_age_preds.append(age_pred)

            for gender_pred in gender_predictions:
                full_gender_preds.append(gender_pred)

            for lab in age_label.detach().cpu().numpy():
                full_age_gts.append(lab)
            for lab in gender_label.detach().cpu().numpy():
                full_gender_gts.append(lab)

            # if numpy.isnan(full_age_preds).any():
            # print(full_age_preds)
                
            age_pre = mean_absolute_error(full_age_gts,full_age_preds)
            age_rmse = numpy.sqrt(mean_squared_error(full_age_gts,full_age_preds))
            gender_pre = accuracy_score(full_gender_gts,full_gender_preds)
            # loss    += nloss.detach().cpu()
            # train_loss_list.append(nloss.item())
            age_top1    += age_pre
            age_rmse_top_1 +=  age_rmse

            gender_top1 += gender_pre

            counter += 1
            index   += stepsize
            # nloss.requires_grad_(True)
            # nloss.backward()
            # self.__optimizer__ = self.Optimizer(filter(lambda p: p.requires_grad, self.parameters()), self.lr,self.weight_decay)
            self.__optimizer__.step()
            telapsed = time.time() - tstart
            tstart = time.time()
            loss = numpy.mean(numpy.asarray(train_loss_list))
            sys.stdout.write("\rProcessing (%d) "%(index))
            sys.stdout.write("Loss %f Age - age_mae %2.3f -  age_rmse  %2.3f- gender - TEER/TAcc %2.3f -- %.2f Hz Time %.3f| %.3f"%(loss, age_top1/counter,age_rmse_top_1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
            sys.stdout.flush()

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n")
        
        return (loss/counter, age_top1/counter, gender_top1/counter)
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def eval_age_network(self, loader):

        self.eval()
        stepsize = loader.batch_size
        counter = 0
        index   = 0
        loss    = 0
        age_top1    = 0     # EER or accuracy
        gender_top1 = 0
        age_rmse_top_1 = 0
        fe_mae_top1 =0
        fe_rmse_top1 =0
        ma_mae_top1 =0
        ma_rmse_top1 =0

        tstart = time.time()
        train_loss_list = list()
        with torch.no_grad():
            for data, age_label,weight,gender_label in loader:

                train_loss_list = []
                full_gender_preds = []
                full_age_preds = []
                full_gender_gts = []
                full_age_gts = []
                data_time = time.time() - tstart

                data = data.transpose(0,1)
                
                feat = []
                for inp in data:
                    outp      = self.__S__.forward(inp.to(self.device))
                    feat.append(outp)

                feat = torch.stack(feat,dim=1).squeeze()

                age_label   = age_label.to(self.device)
                # age_label = age_label.float()
                gender_label = torch.LongTensor(gender_label).to(self.device)

                a_out,g_out = self.__L__.forward(feat)

                # age_loss = F.mse_loss(a_out,age_label)
                # age_loss = self.mse(a_out,age_label)
                weight = weight.to(self.device)
                age_loss = weighted_focal_mse_loss(a_out,age_label,weights = weight)
                # age_loss = self.mse(a_out,age_label)
                gender_loss = self.loss_fun(g_out,gender_label)
                # weight_age = age_loss * age_weight
                # loss = weight_age + gender_loss
                nloss =  self.weight * age_loss + gender_loss

                train_loss_list.append(nloss.item())

                age_predictions = 100 * a_out.detach().cpu().numpy()
                age_label *= 100

                gender_predictions = numpy.argmax(g_out.detach().cpu().numpy(), axis=1)

                for age_pred in age_predictions:
                    full_age_preds.append(age_pred)

                for gender_pred in gender_predictions:
                    full_gender_preds.append(gender_pred)

                for lab in age_label.detach().cpu().numpy():
                    full_age_gts.append(lab)
                for lab in gender_label.detach().cpu().numpy():
                    full_gender_gts.append(lab)

                age_pre = mean_absolute_error(full_age_gts,full_age_preds)
                age_rmse = numpy.sqrt(mean_squared_error(full_age_gts,full_age_preds))
                gender_pre = accuracy_score(full_gender_gts,full_gender_preds)
                fe_mae,fe_rmse,ma_mae,ma_rmse = get_gender_age_mae(full_age_gts,full_age_preds,full_gender_gts,full_gender_preds)
                # loss    += nloss.detach().cpu()
                # train_loss_list.append(nloss.item())
                fe_mae_top1 += fe_mae
                fe_rmse_top1 += fe_rmse
                ma_mae_top1 += ma_mae
                ma_rmse_top1 += ma_rmse
                age_top1    += age_pre
                age_rmse_top_1 += age_rmse

                gender_top1 += gender_pre
                counter += 1
                index   += stepsize
                telapsed = time.time() - tstart
                tstart = time.time()
                loss = numpy.mean(numpy.asarray(train_loss_list))
                sys.stdout.write("\rProcessing (%d) "%(index))
                # sys.stdout.write("Loss %f Age - TEER/TAcc %2.3f - gender - TEER/TAcc %2.3f -- %.2f Hz Time %.3f| %.3f"%(loss, age_top1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
                sys.stdout.write("Loss %f Age - age_mae %2.3f -  age_rmse  %2.3f --female %2.3f %2.3f male %2.3f %2.3f- gender - TEER/TAcc %2.3f -- %.2f Hz Time %.3f| %.3f"%(loss, age_top1/counter,age_rmse_top_1/counter,fe_mae_top1/counter,fe_rmse_top1/counter,ma_mae_top1/counter,ma_rmse_top1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
                sys.stdout.flush()
            sys.stdout.write("\n")

        return (loss/counter, age_top1/counter, gender_top1/counter)
    

    def test_age_network(self, loader):

        self.eval()
        stepsize = loader.batch_size
        counter = 0
        index   = 0
        loss    = 0
        age_top1    = 0     # EER or accuracy
        gender_top1 = 0
        age_rmse_top_1 = 0

        tstart = time.time()
        train_loss_list = list()
        all_age_label = list()
        all_age_predict = list()
        all_gender_label  = list()
        all_gender_predict = list()

        with torch.no_grad():
            for data, age_label,gender_label in loader:

                train_loss_list = []
                full_gender_preds = []
                full_age_preds = []
                full_gender_gts = []
                full_age_gts = []
                data_time = time.time() - tstart

                data = data.transpose(0,1)
                
                feat = []
                for inp in data:
                    outp      = self.__S__.forward(inp.to(self.device))
                    feat.append(outp)

                feat = torch.stack(feat,dim=1).squeeze()

                age_label   = age_label.to(self.device)
                # if index % 5 == 0:
                #     print(age_label)
                # age_label = age_label.float()
                gender_label = torch.LongTensor(gender_label).to(self.device)

                a_out,g_out = self.__L__.forward(feat)

                # # age_loss = F.mse_loss(a_out,age_label)
                # age_loss = self.mse(a_out,age_label)
                # gender_loss = self.loss_fun(g_out,gender_label)
                # # weight_age = age_loss * age_weight
                # # loss = weight_age + gender_loss
                # nloss = age_loss + gender_loss

                # train_loss_list.append(nloss.item())

                age_predictions = 100 * a_out.detach().cpu().numpy()
                if index  == 100:
                    print(age_label)
                    print(age_predictions)
                age_label *= 100
                
                gender_predictions = numpy.argmax(g_out.detach().cpu().numpy(), axis=1)

                for age_pred in age_predictions:
                    full_age_preds.append(age_pred)

                for gender_pred in gender_predictions:
                    full_gender_preds.append(gender_pred)

                for lab in age_label.detach().cpu().numpy():
                    full_age_gts.append(lab)
                for lab in gender_label.detach().cpu().numpy():
                    full_gender_gts.append(lab)

                all_age_label.extend(full_age_gts)
                all_age_predict.extend(full_age_preds)
                all_gender_label.extend(full_gender_gts)
                all_gender_predict.extend(full_gender_preds)

                age_pre = mean_absolute_error(full_age_gts,full_age_preds)
                age_rmse = numpy.sqrt(mean_squared_error(full_age_gts,full_age_preds))
                gender_pre = accuracy_score(full_gender_gts,full_gender_preds)
                # loss    += nloss.detach().cpu()
                # train_loss_list.append(nloss.item())
                age_top1    += age_pre
                age_rmse_top_1 += age_rmse

                gender_top1 += gender_pre
                counter += 1
                index   += stepsize
                # nloss.requires_grad_(True)
                # nloss.backward()
                # self.__optimizer__ = self.Optimizer(filter(lambda p: p.requires_grad, self.parameters()), self.lr,self.weight_decay)
                telapsed = time.time() - tstart
                tstart = time.time()

                sys.stdout.write("\rProcessing (%d) "%(index))
                # sys.stdout.write("Loss %f Age - TEER/TAcc %2.3f - gender - TEER/TAcc %2.3f -- %.2f Hz Time %.3f| %.3f"%(loss, age_top1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
                sys.stdout.write(" Age - age_mae %2.3f -  age_rmse  %2.3f- gender - TEER/TAcc %2.3f -- %.2f Hz Time %.3f| %.3f"%( age_top1/counter,age_rmse_top_1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
                sys.stdout.flush()
            sys.stdout.write("\n")
        numpy.savez("./test_out.npz",all_age_label = all_age_label,all_age_predict = all_age_predict,all_gender_label = all_gender_label,all_gender_predict = all_gender_predict)
        all_age_pre = mean_absolute_error(all_age_label,all_age_predict)
        all_age_rmse = numpy.sqrt(mean_squared_error(all_age_label,all_age_predict))
        all_gender_pre = accuracy_score(all_gender_label,all_gender_predict)

        return all_age_pre,all_age_rmse,all_gender_pre

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path)
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():

            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    def get_result(self,path):
        self.eval()
        with torch.no_grad():
            data = torch.FloatTensor(loadWAV(path,0,True,10)).to(self.device)
            embed = self.__S__.forward(data)
            a_out,g_out = self.__L__.forward(embed)
            a = 100 * a_out.detach().cpu().numpy()
            c = g_out.detach().cpu().numpy()
            d = softmax(c)
            b = numpy.argmax(g_out.detach().cpu().numpy(), axis=1)
        return a[-1],b[-1],d[-1][-1]

def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    #  对损失进行加权
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def get_gender_age_mae(full_age_gts,full_age_preds,full_gender_gts,full_gender_preds):
    female_index = [i for i in range(len(full_gender_gts)) if full_gender_gts[i] == 0]
    female_age_gts = []
    male_age_gts   = []

    female_age_pre = []
    male_age_pre   = []

    for index,gender in enumerate(full_gender_gts):
        if gender == 0:
            female_age_gts.append(full_age_gts[index])
            female_age_pre.append(full_age_preds[index])
        elif gender == 1:
            male_age_gts.append(full_age_gts[index])
            male_age_pre.append(full_age_preds[index])
        else:
            pass

    female_age_mae = mean_absolute_error(female_age_gts,female_age_pre)
    female_age_rmse = numpy.sqrt(mean_squared_error(female_age_gts,female_age_pre))

    male_age_mae = mean_absolute_error(male_age_gts,male_age_pre)
    male_age_rmse = numpy.sqrt(mean_squared_error(male_age_gts,male_age_pre))
    return female_age_mae ,female_age_rmse ,male_age_mae,male_age_rmse

def softmax(x):
    orig_shape=x.shape
    
    if len(x.shape)>1:
        #矩阵
        tmp=np.max(x,axis=1)
        x-=tmp.reshape((x.shape[0],1))
        x=np.exp(x)
        tmp=np.sum(x,axis=1)
        x/=tmp.reshape((x.shape[0],1))
    else:
        #向量
        tmp=np.max(x)
        x-=tmp
        x=np.exp(x)
        tmp=np.sum(x)
        x/=tmp
    return x