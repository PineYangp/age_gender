#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV

class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, use_gpu,**kwargs):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu) 
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).to(self.device)
        self.__S__ = nn.DataParallel(self.__S__)

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).to(self.device)

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(filter(lambda p: p.requires_grad, self.parameters()), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

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

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0
        age_top1    = 0     # EER or accuracy
        gender_top1 = 0

        tstart = time.time()
        
        for data, age_label,gender_label in loader:
            data_time = time.time() -tstart

            data = data.transpose(0,1)

            self.zero_grad()

            feat = []
            for inp in data:
                outp      = self.__S__.forward(inp.to(self.device))
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()

            age_label   = torch.LongTensor(age_label).to(self.device)
            gender_label = torch.LongTensor(gender_label).to(self.device)

            nloss,age_loss,gender_loss,age_pre,gender_pre = self.__L__.forward(feat,age_label,gender_label)

            loss    += nloss.detach().cpu()
            age_top1    += age_pre
            gender_top1 += gender_pre

            counter += 1
            index   += stepsize

            nloss.backward()
            self.__optimizer__.step()

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index))
            sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz Time %.3f| %.3f"%(loss/counter, age_top1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
            sys.stdout.flush()

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n")
        
        return (loss/counter, age_top1/counter, gender_top1/counter)
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=None):
        
        self.eval()
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break

                data = line.split()

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            # inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).to(self.device)
            inp1 = torch.FloatTensor(loadWAV(file, eval_frames, evalmode=True, num_eval=num_eval)).to(self.device)

            ref_feat = self.__S__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].to(self.device)
            com_feat = feats[data[2]].to(self.device)

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

            score = -1 * numpy.mean(dist)

            all_scores.append(score);  
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed))
                sys.stdout.flush()

        print('\n')

        return (all_scores, all_labels, all_trials)

    def eval_age_network(self, loader):

        self.eval()

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0
        age_top1    = 0     # EER or accuracy
        gender_top1 = 0

        tstart = time.time()
        with torch.no_grad():
            for data, age_label,gender_label in loader:
                data_time = time.time() -tstart

                data = data.transpose(0,1)
                feat = []
                for inp in data:
                    outp      = self.__S__.forward(inp.to(self.device))
                    feat.append(outp)

                feat = torch.stack(feat,dim=1).squeeze()

                age_label   = torch.LongTensor(age_label).to(self.device)
                gender_label = torch.LongTensor(gender_label).to(self.device)

                nloss,age_loss,gender_loss,age_pre,gender_pre = self.__L__.forward(feat,age_label,gender_label)

                loss    += nloss.detach().cpu()
                age_top1    += age_pre
                gender_top1 += gender_pre

                counter += 1
                index   += stepsize

                telapsed = time.time() - tstart
                tstart = time.time()

                sys.stdout.write("\rProcessing (%d) "%(index))
                sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz Time %.3f| %.3f"%(loss/counter, age_top1/counter, gender_top1/counter,stepsize/telapsed , data_time,telapsed))
                sys.stdout.flush()

            sys.stdout.write("\n")
        
        return (loss/counter, age_top1/counter, gender_top1/counter)

    def enrollment_dic_kwsTrials(self, listfilename, uttpath, utt2label, save_path, print_interval=10, num_eval=10, eval_frames=None , save_dic=False):
        
        self.eval()
        
        lines       = []
        enroll_utt  = []
        feats       = {}
        tstart      = time.time()
        
        ## Read all lines
        with open(listfilename) as listfile:
            lines = listfile.readlines()



        for line in lines:
            data = line.split()
            enroll_utt.append(data[0])
            enroll_utt.append(data[1])
            enroll_utt.append(data[2])
        set_enroll_utt = list(set(enroll_utt))
        set_enroll_utt.sort()

        ##extract enrollment data embeddings
        for idx, uttid in enumerate(set_enroll_utt):
            with torch.no_grad():
                inp = torch.FloatTensor(loadWAV(uttpath+uttid,0,True,10)).to(self.device)

                embd = self.__S__.forward(inp).cpu()

            feats[uttid] = embd
            telapsed = time.time() - tstart
            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(set_enroll_utt),idx/telapsed,embd.size()[1]));
        feats_np = {}
        for utt in feats:
            feats_np[utt] = feats[utt].numpy()
        if save_dic == True:
            savenpy_path=save_path 
            numpy.save(savenpy_path,feats_np)

        end      = time.time() - tstart
        print("\n total time %.2f"%(end))
        return feats_np



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

