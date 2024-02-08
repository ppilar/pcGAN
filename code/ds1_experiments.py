# -*- coding: utf-8 -*-

import os
import sys
import random
import torch
import numpy as np
from pcGAN.utils import init_random_seeds, set_input_path, init_par


rand_init, s = init_random_seeds(s=0)
path0 = '../results/'


iexp_vec = [0,1,2,3,4,5]

# if len(sys.argv) > 1:
#     iexp_vec = []
#     for j in range(1, len(sys.argv)):
#         iexp = int(sys.argv[j])
#         iexp_vec.append(iexp)
# else:
#     print('error! no experiments selected!')
# print('run experiments:', iexp_vec)



for iexp in iexp_vec:
    print('experiment:', iexp)
    pars = dict()
    init_par(pars, 'model_vec', [1])
    init_par(pars, 'ds_opt', 1)
    init_par(pars, 'Nrun', 3)
    
    init_par(pars, 'lr', 0.0002)
    init_par(pars, 'itmax', 100000) 
    
    
    init_par(pars, 'bs', 128) 
    init_par(pars, 'match_opt', 'KL')   
    init_par(pars, 'omega', 5)
    comment0 = ''
    
    
    if iexp == 0:   
        init_par(pars, 'par_label', 'omega')
        par_vec = [1, 10, 50, 100, 200, 500, 1000]
    if iexp == 1:
        pars['batch_size'] = 64
        init_par(pars, 'par_label', 'fforget')
        par_vec = [0, 0.5, 0.9]        
    if iexp == 2:
        pars['batch_size'] = 128
        init_par(pars, 'par_label', 'fforget')
        par_vec = [0, 0.5, 0.9]        
    if iexp == 3:
        pars['batch_size'] = 256
        init_par(pars, 'par_label', 'fforget')
        par_vec = [0, 0.5, 0.9]
    if iexp == 4:
        pars['match_opt'] = 'abs'
        init_par(pars, 'par_label', 'fforget')
        par_vec = [0.5, 0.9]    
    if iexp == 5:
        pars['match_opt'] = 'JS'
        init_par(pars, 'par_label', 'fforget')
        par_vec = [0.5, 0.9]
    
    
    #run
    for j, par in enumerate(par_vec):
        print('experiment:', iexp, 'run:', j)
        pars[pars['par_label']] = par  
        comment = 'bs' + str(pars['bs']) + '_' + pars['par_label'] + str(par) + comment0
        folder0 = 'ds' + str(pars['ds_opt']) + '_' + pars['match_opt'] + '_' + comment
        input_path = set_input_path(path0, folder0)     
        exec(open('pcGAN.py').read())