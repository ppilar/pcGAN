# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from GAN_pc.results import Results
from GAN_pc.ebms import cEBM
from GAN_pc.utils import *
from GAN_pc.plots import *
from GAN_pc.utils_ebm import *
from GAN_pc.datasets_v2 import *

def set_paths(bs):
    path0 = '../results/ds1_Nd_1_bs_'+str(bs)+'/wave_forms/'
    ppath = path0 + 'plots/'
    fpath = path0 + 'files/'
    fname_ebm = 'ds1_ebms.pk'
    fname_res = 'ds1_results.pk'
    #ajcx = [1, 15, 50]
    cinds_selection = [1, 15, 50]
    return ppath, fpath, fname_res, cinds_selection


#%%


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Ns = 100000
use_cvalue_input = False

ds_opt = 1


#%%
jm = 1
bsvec = [32, 64, 128,256]
cbuf_ges = np.zeros([len(bsvec),101])
pbuf_ges = np.zeros([len(bsvec),6])
for j,bs in enumerate(bsvec):
    print(bs)
    if ds_opt == 1:
        ppath, fpath, fname_res, cinds_selection = set_paths(bs)
        
    with open(fpath+fname_res, 'rb') as f:
        ds, ebms, res = pickle.load(f)
    
    cbuf_ges[j,:] = np.array(res.constraints_ges[jm])[-1,:]# - cKLs_real.numpy()
    pbuf_ges[j,:] = np.array(res.pmetrics_ges[jm])[-1,:]
    

fig, axs = plt.subplots(1,2,figsize=(12,4))
xbuf = range(len(bsvec))
axs[0].plot(xbuf,cbuf_ges)
axs[0].set_xticks(xbuf)
axs[0].set_xticklabels([str(bs) for bs in bsvec])
axs[0].set_xlabel('batch size')
axs[1].plot(pbuf_ges)
axs[1].set_xticks(xbuf)
axs[1].set_xticklabels([str(bs) for bs in bsvec])
axs[1].set_xlabel('batch size')
#%%
#################
################# plot constraints

for jm in [0,1,3]:
    gges = []
    bsvec = [32, 64, 128,256]
    for bs in bsvec:
        print(bs)
        if ds_opt == 1:
            ppath, fpath, fname_res, cinds_selection = set_paths(bs)
            # path0 = '../results/ds1_Nd_1_bs_'+str(bs)+'/'
            # ppath = path0 + 'plots/wave_forms/'
            # fpath = path0 + 'files/wave_forms/'
            # fname_ebm = 'ds1_ebms.pk'
            # fname_res = 'ds1_results.pk'
            # #ajcx = [1, 15, 50]
            # cinds_selection = [1, 15, 50]
            # latent_dim = 5
            # pars = (latent_dim, use_cvalue_input)
            # #ds = wave_forms(Ns=Ns, device=device, pars=pars)
        
        
        
    
        with open(fpath+fname_res, 'rb') as f:
            ds, ebms, res = pickle.load(f)
        gnet = res.gnet_ges[jm][0]
        gges.append(generate_Ns(gnet, 20000, device, gnet.latent_dim).squeeze())
    

    cinds_selection = [0,1,2,3]
    atitle = ['real','fake (32)','fake (64)','fake (128)','fake (256)']
    ds.cinds_selection = cinds_selection#[1,15,30]
    ds.plot_constraints(gges, ebms, spath=ppath+'bs_constraint_comparison', atitle=atitle, all_constraints=False);
    
    
    
    
#%%
#################
################# plot metrics
aj = [1]    
for aj in [[0],[1],[3]]:
    bsvec = [32, 64, 128]#256]
    
    fig, axs = plt.subplots(2,1, figsize=(10,10))
    fig.tight_layout(h_pad=3)
    for bs in bsvec:
        print(bs)
        if ds_opt == 1:
            ppath, fpath, fname_res, cinds_selection = set_paths(bs)

        
        with open(fpath+fname_res, 'rb') as f:
            ds, ebms, res = pickle.load(f)
        
        
        plt.rcParams.update({'font.size': 14})
        eval_it = 500
        mls = ['-','--',':']
        als = ['--','-','-.',':']
        pinds = [0,4,5]#[2,5,8]#[1,4,7]
    
        #aj = [1,0,3]
        for j in aj:
            cbuf = np.array(res.constraints_ges[j])# - cKLs_real.numpy()
            pbuf = np.array(res.pmetrics_ges[j])
            xbuf = eval_it*np.linspace(0,cbuf.shape[0]-1,cbuf.shape[0])    
            
            axplot_metric_trajectory_intervals(axs[0], xbuf, cbuf, ebms.cKLs_real.cpu(), ybounds = [1e-4,1e8], label='jm'+str(j))
            axplot_metric_trajectories(axs[1], xbuf, pbuf[:,pinds], ebms.pKLs_hist_real[pinds].cpu(), np.array(ds.pmetric_names)[pinds], linestyle=mls)
            #axs[1].set_prop_cycle(None)
    
    if ds_opt == 1:
        axs[0].set_ylim(1e-3,1e4)
        axs[1].set_ylim(3e-2,1e2)
    elif ds_opt == 2:
        axs[0].set_ylim(1e-2,3e0)
        axs[1].set_ylim(1e-3,2e1)    
    elif ds_opt == 3:
        axs[0].set_ylim(1e-3,1e2)
        axs[1].set_ylim(5e-3,3e1)
    
    # Get the handles and labels of the lines
    handles, labels = axs[0].get_legend_handles_labels()
    new_handles = [handles[0], handles[2], handles[4], handles[5]]
    new_labels = ['pcGAN (32)', 'pcGAN (64)', 'pcGAN (128)','real data']
    axs[0].legend(new_handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    leg = axs[1].legend(np.array(ds.pmetric_names)[pinds])
    for j in range(len(pinds)):
        leg.legendHandles[j].set_color('k')
        
    plt.savefig(ppath + 'bs_metrics_summary.pdf')