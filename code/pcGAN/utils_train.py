# -*- coding: utf-8 -*-

import os

import random
import shutil
import time
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
from .Nets import *
from .datasets import *
from .utils import *
from .plots import plot_losses, plot_metric_trajectories, plot_cebm
from .datasets import *
from .cebm import *



#initialize dataset according to ds_opt
def initialize_ds(ds_opt, device):
    latent_dim = 100 if ds_opt == 2 else 5
    use_cvalue_input = False
    
    pars = (latent_dim, use_cvalue_input)
    if ds_opt == 1:
        Ns = 100000
        ds = wave_forms(Ns=Ns, device=device, pars=pars)
    if ds_opt == 2:
        ds = CAMELS(device=device, pars=pars)
    if ds_opt == 3:
        ds = IceCube_wave_forms(device=device,pars=pars)
        
    return ds


#create directories to store results and plots
def create_dirs(input_path, ds_name):
    dpath = input_path + ds_name + '/'
    rpath = dpath + 'files/'
    ppath0 = dpath + 'plots/'
    
    
    check_dirs(input_path, dpath)    
    check_dirs(input_path, ppath0)
    check_dirs(input_path, rpath)

    return rpath, ppath0


#initialize cEBM: train, plot and determine fsig
def initialize_cebm(ds, ds_opt, bs, Neval, ppath0, rpath, load_cebm, device, plot=True):    
    if load_cebm == 0:
        tebm = time.time()
        cebm = cEBM(ds.ds_name, ds.constraints[:,:ds.Nebm], ds.constraint_names, device, Nit=ds.Nit_ebm, cut_tails = ds.cut_tails)
        if plot: plot_cebm(cebm, ds.constraints[:,:ds.Nebm], ds.constraint_names, ppath0)
        cebm.calculate_real_data_metrics(ds, bs, Neval=Neval)
        print('tEBM:'+str(tebm - time.time()))
            
    else:
        with open(rpath+'ds'+str(ds_opt)+'_results.pk', 'rb') as f:
            ds, cebm, _ = pickle.load(f)
        if plot: plot_cebm(cebm, ds.constraints[:,:ds.Nebm], ds.constraint_names, ppath0)
        if bs not in cebm.abs:
            cebm.calculate_real_data_metrics(ds, bs, Neval=Neval)
        else:
            cebm.set_batch_size(bs)

    return cebm, ds


#print loss values and add to results
def print_and_update(res, jm, epoch, itges, ti, dloss0, dloss_GP, gloss0, loss_kl):
    if itges > 1:
        print("\r",
              'ep', epoch,
              'it', itges,
              'ld', round(np.array(res.losses[jm][0][-50:]).mean(),5),
              'ld2', round(np.array(res.losses[jm][1][-50:]).mean(),5),
              'lg', round(np.array(res.losses[jm][2][-50:]).mean(),5),
              'lg2', round(np.array(res.losses[jm][3][-50:]).mean(),5),
              'ti', round(time.time() - ti, 3),
              end="")
    
    res.losses[jm][0].append(dloss0.item())
    res.losses[jm][1].append(dloss_GP.item())
    res.losses[jm][2].append(gloss0.item())
    res.losses[jm][3].append(loss_kl.item())
    return res

#evaluate constraint fulfillment and performance metrics; plot intermediate results
def plot_and_evaluate(ds, res, ebms, jm, itges, dnet, gnet, gsin, gcval, ppath, mstr, device, ieval, Neval):
    teval = time.time()
    gnet.eval()
    plot = itges%res.plot_it == 0 and itges > 0
    
    
    #plot samples (with same random input)
    if plot:
        gsamples = gnet(gsin)
        dscores = dnet((gsamples, gcval))
        ds.plot_samples((gsamples, dscores), ppath + 'fake_it'+str(itges))
        
        dsamples = ds.data[:16,:]
        dscores0 = dnet((dsamples, gcval))
        ds.plot_samples((dsamples, dscores0), ppath + 'true_it'+str(itges))
        
    
    
    #generate bigger dataset to monitor statistics
    gges = generate_Ns(gnet, Neval, device, gnet.latent_dim).squeeze()
    if plot:
        ds.plot_summary(gges, ppath+'summary')
    
    pbuf = ds.plot_pmetrics(gges, ppath, plot=plot)
    cbuf = ds.plot_constraints(gges, ebms, ppath, plot=plot)
    
    cval_ges = ds.calculate_constraints(gges)                
    kls_ges = calculate_multiple_KL(ebms, cval_ges[:,:ds.Nebm], fsig=ebms.fsig_best).tolist()
    
    res.constraints_ges0[jm].append(cbuf)
    res.constraints_ges[jm].append(kls_ges)
    res.pmetrics_ges[jm].append(pbuf)
    
    #plot loss curves
    if plot:
        plot_losses(res.losses[jm], ppath+mstr+'losses.pdf')
        plot_metric_trajectories(res.constraints_ges[jm], res.pmetrics_ges[jm], (ebms.cKLs_real, ebms.pKLs_hist_real), ds.constraint_names, ds.pmetric_names, res.eval_it, ppath + 'KL')
        plot_metric_trajectories(res.constraints_ges0[jm], res.pmetrics_ges[jm], (ebms.cKLs_hist_real, ebms.pKLs_hist_real), ds.constraint_names, ds.pmetric_names, res.eval_it, ppath + 'hist')

    
    gnet.train()
    ieval += 1
    
    return res, ieval