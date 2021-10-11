'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-28 16:58:20
LastEditors: yp
LastEditTime: 2021-10-10 10:12:53
'''
#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
# import yaml
import numpy
import pdb
import torch
import glob
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from AgeNet import SpeakerNet
from Age_dataloader import get_data_loader,get_test_loader

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=300,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=128,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=15000,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')

parser.add_argument('--augment',        type=bool,  default=True,  help='Augment input')

#  标签平滑，  固定
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
parser.add_argument('--reweight', type=str, default='sqrt_inv', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')


## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=50,    help='Maximum number of epochs')

parser.add_argument('--trainfunc',      type=str,   default="amsoftmax",     help='Loss function')

parser.add_argument('--newtrainfunc',      type=str,   default="no_sigmoid",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')

parser.add_argument('--lr',             type=float, default=0.01,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.2,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=1e-4,      help='Weight decay in the optimizer')

parser.add_argument('--age_Classes',       type=int,   default=1,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--gender_Classes',       type=int,   default=2,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--nClasses',       type=int,   default=1167,   help='Number of speakers in the softmax layer, only for softmax-based losses')
## speaker model Load and save
parser.add_argument('--start_epoch',    type=int,   default=1,     help='Initial the number of start epoch')
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_interval',  type=int,   default=100,     help='Save the model per epochs')
parser.add_argument('--save_path',      type=str,   default="./data/exp1", help='Path for model and logs')

## age model Load and save
parser.add_argument('--age_start_epoch',    type=int,   default=1,     help='Initial the number of start epoch')
parser.add_argument('--age_initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--age_save_interval',  type=int,   default=2,     help='Save the model per epochs')
parser.add_argument('--age_save_path',      type=str,   default="/store2/home2/yangp/AGE/spk/age_model_save/", help='Path for model and logs')
 
## Training and test data 
parser.add_argument('--train_list',     type=str,   default="",     help='Train list')
parser.add_argument('--test_list',      type=str,   default="",     help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="./metadata/train_age.txt", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="./metadata/eval_age.txt", help='Absolute path to the test set')

parser.add_argument('--musan_path',     type=str,   default="/data/musan", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/data/rirs_noises/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="ResNetSE34v2",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer ')

parser.add_argument('--use_gpu', type=str, default=1, help="gpu")
parser.add_argument('--weight', type=float, default=1, help="age loss weight")
parser.add_argument('--tm', type=float, default=1, help="age predict para")
## For test only

parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

parser.add_argument('--flag', type=int, default=0, help="0 no pretrain,1 spk pretrain, 2 spk,and common pretrain")

parser.add_argument('--loss_type', type=str, default="focal_mse_loss", help="select loss function: weight_mse_loss,mse_loss,focal_mse_loss")

parser.add_argument('--freequze', type=str, default="", help="  fintune part ")

parser.add_argument('--weight_type', type=str, default="error", help=" loss function")

parser.add_argument('--flag_vad', type=bool, default=False, help="use low pass filter")

parser.add_argument('--train_mode', type=str, default="lr_fintune", help="train method [lr_fintune, freequze]")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu) 
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

## Parse YAML

def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

class train():

    def __init__(self,model_save_path,flag = 0):

        self.it          = 1
        self.prevloss    = float("inf")
        self.sumloss     = 0

        self.freequze = args.freequze

        self.model_save_path = model_save_path
        if not(os.path.exists(self.model_save_path)):
            os.makedirs(self.model_save_path)
        self.s = SpeakerNet(**vars(args))

        if flag == 0:
            pass

        if flag == 1:

            self.s.loadParameters(args.initial_model)
            LossFunction = importlib.import_module('loss.'+args.newtrainfunc).__getattribute__('LossFunction')

            self.s.__L__ = LossFunction(nOut = args.nOut, age_Classes = 1, gender_Classes = 2,margin=0.3, scale=15).to(device)
            Optimizer = importlib.import_module('optimizer.'+args.optimizer).__getattribute__('Optimizer')

            if args.train_mode == "lr_fintune":
        
                pre_layer = list(map(id,self.s.__L__.parameters()))
                base_pram = filter(lambda p:id(p) not in pre_layer,self.s.parameters())
                
                par =  [{'params': base_pram}, 
                        {'params': self.s.__L__.parameters(), 'lr': args.lr * 10}]

                self.s.__optimizer__ = Optimizer(par, args.lr,args.weight_decay)

            elif args.train_mode == "freequze":

                for name, param in self.s.named_parameters():
                    if "__S__" in name:
                        param.requires_grad = False

                self.s.__optimizer__ = Optimizer(filter(lambda p: p.requires_grad, self.s.parameters()), args.lr,args.weight_decay)

            else:
                self.s.__optimizer__ = Optimizer(filter(lambda p: p.requires_grad, self.s.parameters()), args.lr,args.weight_decay)   

            Scheduler = importlib.import_module('scheduler.'+args.scheduler).__getattribute__('Scheduler')
            self.s.__scheduler__, self.lr_step = Scheduler(self.s.__optimizer__, test_interval = 1,  lr_decay = args.lr_decay)
            
        
        if flag == 2:

            self.s.loadParameters(args.initial_model)
            LossFunction = importlib.import_module('loss.'+args.newtrainfunc).__getattribute__('LossFunction')

            self.s.__L__ = LossFunction(nOut = args.nOut, age_Classes = 1, gender_Classes = 2,margin=0.3, scale=15).to(device)
            
            Optimizer = importlib.import_module('optimizer.'+args.optimizer).__getattribute__('Optimizer')
            self.s.__optimizer__ = Optimizer(filter(lambda p: p.requires_grad, s.parameters()), args.lr,args.weight_decay)   

            Scheduler = importlib.import_module('scheduler.'+args.scheduler).__getattribute__('Scheduler')
            self.s.__scheduler__, self.lr_step = Scheduler(self.s.__optimizer__, test_interval = 1,  lr_decay = args.lr_decay)

        for ii in range(0,self.it-1):
            self.s.__scheduler__.step()


    def train(self):

        trainLoader = get_data_loader(args.train_list, **vars(args))

        evalLoader = get_test_loader(args.test_list,**vars(args))

        min_mae = 1000

        min_rmse = 1000

        while(1):
            
            clr = [x['lr'] for x in self.s.__optimizer__.param_groups]

            print(time.strftime("%Y-%m-%d %H:%M:%S"), self.it, "Training %s with LR %f..."%(args.model,max(clr)))

            loss, age_mae,age_rmse,f_mae,f_rmse,m_mae,m_rmse, g_acc = self.s.train_age_network(loader=trainLoader)

            loss, age_mae,age_rmse,f_mae,f_rmse,m_mae,m_rmse, g_acc = self.s.eval_age_network(loader=evalLoader)

            if age_mae < min_mae or age_rmse < min_rmse:

                self.s.saveParameters(self.model_save_path+"/model%09d"%self.it + "age_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_gender_{:.4f}.model".format(age_mae,age_rmse,f_mae,f_rmse,m_mae,m_rmse, g_acc))

                if age_mae < min_mae:
                    min_mae = age_mae
                
                if age_rmse < min_rmse:
                    min_rmse = age_rmse

            if self.it >= args.max_epoch:

                quit()

            self.it+=1


if __name__ == "__main__":

    run = train(args.save_path,flag = args.flag)

    run.train()





