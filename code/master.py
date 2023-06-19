# -*- coding: utf-8 -*-
#use this file to loop over various settings

import os
import random
import torch
import numpy as np
from pcGAN.utils import init_random_seeds, set_input_path


rand_init, s = init_random_seeds(s=0)
path0 = '../results/'

#loop over bs
ds_opt = 1 #choose dataset; 1 ... synthetic dataset; 2 ... Tmaps; 3 ... IceCube-Gen2 signals
bs_vec = [256] #batch sizes
Nd_vec = [1] #number of discriminator iterations
for j, bs in enumerate(bs_vec):
    for jNd, Nd in enumerate(Nd_vec):
        folder0 = 'ds' + str(ds_opt) + '_Nd_' + str(Nd) + '_bs_' + str(bs)
        input_path = set_input_path(path0, folder0)    
        exec(open('pcGAN.py').read())


