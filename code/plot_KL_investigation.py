# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

#from pcGAN.cebm import calculate_multiple_KL
from pcGAN.utils import *
from pcGAN.datasets import *
from pcGAN.utils_summary_plots import *
from pcGAN.utils_train import initialize_ds, initialize_ptrue_rep
from pcGAN.representations.rKDE import rKDE
from pcGAN.representations.utils_representations import calculate_KLs_for_fsig






#%%   
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


pars = dict()
pars['ds_opt'] = 1
pars['load_ds'] = 1
pars['load_ptrue_rep'] = 1
pars['ptrue_rep'] = 'KDE'
pars['match_opt'] = 'KL'
pars['Neval'] = 5000
pars['include_history'] = 0



ppath0 = ''
fpath0 =  '../results/datasets/' 
ds = initialize_ds(pars, device, fpath0)
ptrue_rep = initialize_ptrue_rep(ds, pars, ppath0, fpath0, update_fsig = False, plot=False)
pars['match_opt'] = 'KL'
ptrue_rep2 = initialize_ptrue_rep(ds, pars, ppath0, fpath0, update_fsig = False, plot=False)
#%% calculate KL values

#initialize parameters
Navg = 50
Nsig = 100
fsig_vec = np.logspace(-1,2,Nsig)
bsvec = [32,64,256]
Nbs = len(bsvec)
lvec = [str(j) for j in bsvec]
lvec = ['bs='+ str(j) for j in lvec]
KLs0 = torch.zeros(ptrue_rep.Nc, Nbs, Nsig, Navg)
KLs = torch.zeros(ptrue_rep.Nc, Nbs, Nsig, Navg)

#%%


for jbs, bs in enumerate(bsvec): #loop over batch sizes 
    print('bs:',str(bs))
    
    KLs[:, jbs, :, :] = calculate_KLs_for_fsig(ds, ptrue_rep, 'abs', bs, Nsig = Nsig, Navg = Navg)[0]
    KLs0[:,jbs, :, :] = calculate_KLs_for_fsig(ds, ptrue_rep2, 'KL', bs, Nsig = Nsig, Navg = Navg)[0]
    


#%% determine fsigbest; fsig_best0 corresponds to the case without cutting tails
KLs0_min, iKLs0_min = KLs0.mean(-1).min(-1)
KLs_min, iKLs_min = torch.abs(KLs.mean(-1)).min(-1)
fsig_best0 = fsig_vec[iKLs0_min]
fsig_best = fsig_vec[iKLs_min]



#%% summary plot
ppath = '../plots/'

if pars['ds_opt'] == 1:
    ajcx = [0, 15, 30]
    jcx_long_tails = 15
    KL_ylim = [1e-2, 5e1]
if pars['ds_opt'] == 3:
    ajcx = [0,1]
    KL_ylim = [1e-1, 1e1]


labels = [ds.constraint_names[j] for j in ajcx]
if len(ajcx) == 3:
    fig, axs = plt.subplots(2,4, figsize=(22,9))
if len(ajcx) == 2:
    fig, axs = plt.subplots(2,3, figsize=(16.5,9))
fig.tight_layout(h_pad=3)
plt.rcParams['font.size'] = 14
for j, jcx in enumerate(ajcx):
    yl_opt = True if j == 0 else False
    axplot_KLdiv(axs[0,j], fsig_vec, KLs[jcx], labels[j], lvec, ylim = KL_ylim, yl_opt = yl_opt)
    axplot_Gmix(axs[1,j], ds, ptrue_rep, bsvec, fsig_best, jcx, crange_opt = 'plot')

axplot_fsig_best(axs[0,len(ajcx)], fsig_best, lvec)
axs[1,len(ajcx)].set_visible(False)
plt.savefig(ppath+'ds'+str(pars['ds_opt'])+'_KL_summary'+'.pdf', bbox_inches='tight')