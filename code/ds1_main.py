# -*- coding: utf-8 -*-

import os
import random
import torch
import numpy as np
from pcGAN.utils import init_random_seeds, set_input_path, init_par


rand_init, s = init_random_seeds(s=0)
path0 = '../results/'

for model in [0, 1, 3, 4, 5]:
    pars = dict()
    init_par(pars, 'model_vec', [model])
    init_par(pars, 'ds_opt', 1)
    init_par(pars, 'Nrun', 3)
    
    init_par(pars, 'itmax', 100000)
    init_par(pars, 'bs', 256)
    
    init_par(pars, 'lr', 0.0002)
    if model == 3:
        init_par(pars, 'omega', 1)
    else:
        init_par(pars, 'omega', 5)
        
    
    init_par(pars, 'par_label', 'model')
    par_vec = [model]
        
        
    
    #run
    for j, par in enumerate(par_vec):
        pars[pars['par_label']] = par  
        comment = '_bs' + str(pars['bs']) + '_' + pars['par_label'] + str(par)
        folder0 = 'ds' + str(pars['ds_opt']) + comment
        input_path = set_input_path(path0, folder0)     
        exec(open('pcGAN.py').read())
        