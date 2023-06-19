# -*- coding: utf-8 -*-
import os

import random
import shutil
import time
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

#initialize random seeds
def init_random_seeds(s=False):
    if type(s) == bool:
        s = s = np.random.randint(42*10**4)
        
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    
    rand_init = 1
    return rand_init, s

#create string leading to folder; also create folder
def set_input_path(path0, folder0, replace=True):
    input_path = path0 + folder0 + '/'
    check_dirs(path0, input_path, replace=True)    
    print(input_path)
    return input_path


#######
####### 


#check if input_path exists and create if it does not
def check_dirs(path0, input_path, replace=False):
    
    if not os.path.exists(input_path):
        os.mkdir(input_path)    
    if not os.path.exists(input_path + 'input.py') or replace == True:
        shutil.copyfile(path0 + 'input.py', input_path + 'input.py')


#get random numbers for generated batch
def get_randn(bs, device, latent_dim):
    return torch.randn(bs*latent_dim, device=device).reshape(bs, latent_dim).float()

#gnereate Ns samples from GAN
def generate_Ns(gnet, Ns, device='cpu', latent_dim = 5):
    N0 = 1000
    Nbuf = 0
    
    i = 0
    while (Nbuf < Ns):
        Ni = min(N0, Ns-Nbuf)
        gin = get_randn(Ni, device, latent_dim)
        gbatch = gnet(gin).detach()
        if i == 0:
            gges = gbatch
        else:
            gges = torch.cat((gges,gbatch))
        i += 1
        Nbuf = Nbuf + Ni
    return gges

#update weights used for sampling constraints
def update_wvec(aJ, f=0.1):
    wexp = 1
    buf = (aJ - aJ.min() + (aJ.max() - aJ.min())*f + 1e-4)**wexp
    wvec = buf/buf.sum()
    return wvec

