# -*- coding: utf-8 -*-

import os
from datetime import datetime

import sys
import random
import shutil
import time
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

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

#print parameters
def print_parameters(pars, fpath):
    with open(fpath, 'w') as f:
        for var in pars.keys():
            f.write(var+':'+str(pars[var])+'\n')
        f.write('\n\n')

#initialize parameter value, if no value has been assigned previously
def init_par(pars, key, val):
    if not key in pars: pars[key] = val
    
#update parameter value to new value
def update_par(pars, key, val, it=None, rpath=None):
    pars[key] = val
    if not it is None:
        print_parameter_update(key, val, it, rpath)
        
#write parameter update to readme
def print_parameter_update(key, val, it, rpath):
    freadme = rpath + '/../readme.txt'
    with open(freadme, 'a+') as f:
        f.write('it' + str(it) + '_' + key + ':' + str(val) + '\n')


#initialize parameters with standard values
#if no other values have been specified up to this point (e.g. in input.py)
def initialize_standard_pars(pars):
    #commonly changed parameters
    init_par(pars, 'model_vec', [1])  
    #0 ... (...)GAN, 1 ... GAN + KL, 2 ... KL, 3 ... covariance (Wu et al.), 4 ... WGAN-GP, 5 ... SNGAN
    #6 ... WGAN-GP + KL, 7 ... SNGAN + KL
    init_par(pars, 'ds_opt', 1) #0 ... sum of numbers; 1 ... wave forms; 2 ... CAMELS; 3 ... IceCube
    init_par(pars, 'load_ds', 1) #0 ... create new data; 1 ... load data
    init_par(pars, 'load_ptrue_rep', 1) #0 ... determine new ptrue representation; 1 ... load existing

    init_par(pars, 'itmax', 100000) #150000
    init_par(pars, 'Nrun', 1)
    init_par(pars, 'bs', 128) #batch size
    init_par(pars, 'omega', 1) #weighting factor
    init_par(pars, 'fforget', 0.9) #epsilon in paper
    
    #standard values of parameters, only to be change when certain
    init_par(pars, 'Nd', 1) #discriminator updates per iteration
    init_par(pars, 'Neval', 6000) #number of evaluation points

    init_par(pars, 'Njkl', -1)  #how many constraints to consider per iteration (-1 ... all constraints)
    init_par(pars, 'match_opt', 'KL')
    init_par(pars, 'delta_opt', False) # True: metric towards real data value; False: towards zero
    #init_par(pars, 'weight_loss_kl', 1)
    init_par(pars, 'constraint_weight_opt', 1)

    init_par(pars, 'pfake_opt', 'KDE')  # options: 'KDE', 'EBM'
    init_par(pars, 'ptrue_rep', 'KDE')

    init_par(pars, 'include_history', 1) #0 ... none; 1 ... exp decay; 2 ... N recent steps
    
    #init_par(pars, 'ebm_cut_tails', False)
    init_par(pars, 'D_batches_together', True)
    init_par(pars, 'par_label', 'none')
    init_par(pars, 'sn_dloss_max', 50)

    #standard values, to be changed only according to model
    init_par(pars, 'GAN_opt', 1) #0 ... GAN, 1 ... WGAN, 2 ... WGAN-GP, 3 ... SNGAN
    init_par(pars, 'jm', -1)
    init_par(pars, 'use_dloss', True)
    init_par(pars, 'use_gloss', True)
    init_par(pars, 'use_pc_loss', False)
    init_par(pars, 'use_Wu_loss', False)


#define a new model given keys and corresponding vals
def define_model(keys, vals, res, pars, it=None, rpath=None):
    mname = ''
    for j in range(len(keys)):
        if j > 0 :
            mname = mname + '-'
        key = keys[j]
        val = vals[j]
        add_to_initial(pars, res, key)
        update_par(pars, key, val)
        mname = mname + key + ' - ' + str(val)
    return mname

#set parameter values to those corresponding to chosen model
def update_model_pars(jm, res, pars, rpath='', it=0):
    pars['jm'] = jm
    if jm == 0:
        mname = 'GAN'
        define_model(
            [],
            [],
            res, pars)
    if jm == 1:
        mname = 'GAN + pc'
        define_model(
            ['use_pc_loss'],
            [True],
            res, pars)
    if jm == 2:
        mname = 'pc'
        define_model(
            ['use_dloss', 'use_gloss', 'use_pc_loss'],
            [False, False, True],
            res, pars)  
    if jm == 3:
        mname = 'GAN + Wu'
        define_model(
            ['use_Wu_loss', 'use_pc_loss'],
            [True, False],
            res, pars)
    if jm == 4:
        mname = 'WGAN-GP'
        define_model(
            ['GAN_opt'],
            [2],
            res, pars)
    if jm == 5:
        mname = 'SNGAN'
        define_model(
            ['GAN_opt'],
            [3],
            res, pars)
    if jm == 6:
        mname = 'pcGAN - GP'
        define_model(
            ['use_pc_loss','GAN_opt'],
            [True, 2],
            res, pars)
    if jm == 7:
        mname = 'pcGAN - SN'
        define_model(
            ['use_pc_loss','GAN_opt'],
            [True, 3],
            res, pars)
        
    res.mname_vec[jm] = mname
    
#keep track of parameter values before choosing model
def add_to_initial(pars, res, key):
    res.initial_pars[key] = pars[key]
    
#reset parameter values to initial values
def reset_to_initial(pars, res):
    for key in res.initial_pars:
        update_par(pars, key, res.initial_pars[key])

#write to same line in console
def write_and_flush(msg):
    sys.stdout.write('\r'+msg)
    sys.stdout.flush()
    
#get dataset name
def get_ds_name(ds_opt):
    if ds_opt == 1:
        ds_name = 'wave_forms'
    if ds_opt == 3:
        ds_name = 'IceCube'
    return ds_name
    
#######
####### 


#check if input_path exists and create if it does not
def check_dirs(path0, input_path, replace=False, copy_input = True):    
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if copy_input == True:
        if not os.path.exists(input_path + 'input.py') or replace == True:
            shutil.copyfile(path0 + 'input.py', input_path + 'input.py')

def initialize_writer(log_dir, comment0 = "", comment = ""):
    #comment=""
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, comment0 + "_" + current_time + "_" + comment)
    return SummaryWriter(log_dir = log_dir)

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

#sample constraint according to loss_values and update weights
def sample_and_update_weights(delta_cs, awc, aavg):
    Nc = delta_cs.shape[0]
    jc = np.random.choice(range(Nc), 1, p=np.array(awc)).item()
    buf = delta_cs[jc]
    aavg[jc] = (aavg[jc] + buf.item())/2
    awc = update_wvec(aavg)
    
    return jc, buf, aavg, awc

#update weights used for sampling constraints
def update_wvec(aJ, f=0.1):
    wexp = 1
    buf = (aJ - aJ.min() + (aJ.max() - aJ.min())*f + 1e-4)**wexp
    wvec = buf/buf.sum()
    return wvec

#constraint violations evaluated between histograms
def constraint_error_hist(vals0, vals1, bins, match_opt):
    lbin = np.diff(bins)
    inz = np.where(vals0 > 0)[0]

    if match_opt == 'KL':
        res = np.sum(vals0[inz]*np.log(vals0[inz]/(vals1[inz]+1e-9))*lbin[inz])
    else:
        res = np.sum(np.abs(vals0[inz] - vals1[inz])*lbin[inz])
    
    return res

#calculate log probability of Gaussian
def log_prob_gaussian(x, mean, std):
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi).float()) - torch.log(std) - 0.5 * ((x - mean) / std)**2
    return log_prob


#calculate constraint violations according to chosen match_opt
def calculate_dist_loss(pvec_true, pvec_fake, xvecs, match_opt = 'KL'):
    
    if match_opt == 'KL':
        pvec = pvec_true      
        rvec = pvec*(torch.log((pvec+1e-9)/(pvec_fake + 1e-9)))
        rges = torch.trapz(rvec, xvecs, dim=1)
            
    if match_opt == 'JS':
        pvec = pvec_true
        rvec1 = pvec*(torch.log((pvec+1e-9)/(pvec_fake + 1e-9)))
        rges1 = torch.trapz(rvec1, xvecs, dim=1)
        
        rvec2 = pvec_fake*(torch.log((pvec_fake+1e-9)/(pvec + 1e-9)))
        rges2 = torch.trapz(rvec2, xvecs, dim=1)
        
        rges = 0.5*(rges1 + rges2)
        
        
    if match_opt == 'abs':
        rvec = torch.abs(pvec_true - pvec_fake)
        rges = torch.trapz(rvec, xvecs, dim=1)
    
        
    return rges