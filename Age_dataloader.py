'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-13 15:14:26
LastEditors: yp
LastEditTime: 2021-10-10 10:16:46
'''
#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import numpy as np
import random
import pdb
import os
import threading
import time
import math
import glob
import librosa
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import convolve1d
from torch.utils.data import Dataset, DataLoader


def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def get_all_type_paths(file_dir, type_flag):
    """
    获取指定文件夹及其子文件夹下的所有指定类型文件路径
    :param file_dir: 文件夹地址（str）
    :param type_flag: 文件类型
    :return: 地址列表（list）
    """
    _paths = []
    for dir, _, file_base_name in os.walk(file_dir):
        for file in file_base_name:
            if file.endswith(type_flag):
                _paths.append(os.path.join(dir, file))
    return _paths

def get_gender_label(gender):
    if gender == "Female" or gender == "female":
        return 0
    else:
        return 1

def get_age_class(age):
    if age < 20:
        return 0
    elif 20 <= age < 30:
        return 1
    elif 30 <= age < 40:
        return 2
    elif age >= 40:
        return 3
    else:
        return False
     

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)
    if sample_rate != 16000:
        audio, sample_rate = librosa.load(filename, sr=16000)
    if len(audio.shape)==2:
        audio = audio[...,0]
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        try:
            shortage    = max_audio - audiosize + 1 
            audio       = numpy.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]
        except Exception as e:
            print(e,filename)

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
            
    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,10],'speech':[10,15],'music':[5,10]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio


    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        fs, rir     = wavfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


class age_loader(Dataset):
    def __init__(self, dataset_file_name,  augment, musan_path, rir_path, max_frames, train_path,reweight,lds,lds_kernel,lds_ks,lds_sigma):
        self.augment = augment
        if augment:
            self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)
        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames

        with open(self.dataset_file_name,"r",encoding = "utf-8") as fid:
            lines = fid.readlines()
        lines = [line.rstrip() for line in lines]
        random.shuffle(lines)

        self.data_list = []
        self.age_labels = []
        self.gender_labels = []
        for index,line in enumerate(lines):
            spk,wav,age,gender = line.split()
            gender_label = get_gender_label(gender)
            
            age_label = float(age) / 100
            self.gender_labels.append(gender_label)
            self.data_list.append(wav)
            self.age_labels.append(age_label)
            if index == 1:
                print(age_label)

        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)


    def __getitem__(self, index):

        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
        
        if self.augment:
            augtype = random.randint(0,4)
            if augtype == 1:
                audio     = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio   = self.augment_wav.additive_noise('music',audio)
            elif augtype == 3:
                audio   = self.augment_wav.additive_noise('speech',audio)
            elif augtype == 4:
                audio   = self.augment_wav.additive_noise('noise',audio)

        sample = {'feat': audio,
                  'age_label': self.age_labels[index],
                  'gender_label':self.gender_labels[index],
                  "weight": self.weights[index]}
    
        return sample

    def __len__(self):
        return len(self.data_list)

    def _prepare_weights(self, reweight, max_target=80, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.age_labels
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def speech_collate(batch):
    age_targets = []
    gender_targets = []
    specs = []
    weights = []
    for sample in batch:
        specs.append(sample['feat'])
        age_targets.append(sample['age_label'])
        gender_targets.append(sample['gender_label'])
        weights.append(sample["weight"])
    return torch.FloatTensor(specs), torch.FloatTensor(age_targets),torch.FloatTensor(weights), gender_targets


def get_data_loader(dataset_file_name, batch_size, augment, musan_path, rir_path, max_frames,nDataLoaderThread, train_path ,reweight,lds,lds_kernel,lds_ks,lds_sigma,**kwargs):
    
    train_dataset = age_loader(dataset_file_name, augment, musan_path, rir_path, max_frames, train_path,reweight,lds,lds_kernel,lds_ks,lds_sigma)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn = speech_collate
    )
    
    return train_loader


def get_data_test_loader(dataset_file_name, batch_size, augment, musan_path, rir_path, max_frames,nDataLoaderThread, nPerSpeaker, train_path,reweight,lds,lds_kernel,lds_ks,lds_sigma, **kwargs):
    
    train_dataset = age_test_loader(dataset_file_name, augment, musan_path, rir_path, max_frames, train_path,reweight,lds,lds_kernel,lds_ks,lds_sigma)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn = speech_collate
    )
    
    return train_loader
