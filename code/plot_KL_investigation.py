# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from pcGAN.utils import init_random_seeds
from pcGAN.cebm import calculate_multiple_KL
from pcGAN.datasets import *
from pcGAN.utils_summary_plots import *


#%%   

rand_init, s = init_random_seeds(s=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds_opt = 1 #choose dataset: 1 ... synthetic, 2 ... Tmaps, 3 ... IceCube-Gen2
ds, net_ebm, _, ajcx, jcx_long_tails, KL_ylim, _, _, ppath = initialize_cebm_ds(ds_opt) #load trained cebm + corresponding dataset

#%% calculate KL values

#initialize parameters
Navg = 50
Nsig = 100
fsig_vec = np.logspace(-1,2,Nsig)
bsvec = [32,64,256]
Nbs = len(bsvec)
lvec = [str(j) for j in bsvec]
lvec = ['bs='+ str(j) for j in lvec]
KLs0 = torch.zeros(net_ebm.Nc, Nbs, Nsig, Navg)
KLs = torch.zeros(net_ebm.Nc, Nbs, Nsig, Navg)

for jbs, bs in enumerate(bsvec): #loop over batch sizes 
    print('bs:',str(bs))
    for jN in range(Navg): #loop over different minibatches
        sys.stdout.write('\r'+str(jN))
        sys.stdout.flush()
        mb, cmb = sample_mb(ds, bs)
        for jsig, fsig in enumerate(fsig_vec): #loop over different  values of fsig 
            KLs[:,jbs, jsig, jN] = calculate_multiple_KL(net_ebm, cmb[:,:], fsig = fsig).detach().cpu()
            KLs0[:,jbs, jsig, jN] = calculate_multiple_KL(net_ebm, cmb[:,:], fsig = fsig, cut_tails=False).detach().cpu()


#%% determine fsigbest; fsig_best0 corresponds to the case without cutting tails
KLs0_min, iKLs0_min = KLs0.mean(-1).min(-1)
KLs_min, iKLs_min = torch.abs(KLs.mean(-1)).min(-1)
fsig_best0 = fsig_vec[iKLs0_min]
fsig_best = fsig_vec[iKLs_min]



#%% summary plot

labels = [ds.constraint_names[j] for j in ajcx]
fig, axs = plt.subplots(2,4, figsize=(22,9))
fig.tight_layout(h_pad=3)
plt.rcParams['font.size'] = 14
for j, jcx in enumerate(ajcx):
    yl_opt = True if j == 0 else False
    axplot_KLdiv(axs[0,j], fsig_vec, KLs[jcx], labels[j], lvec, ylim = KL_ylim, yl_opt = yl_opt)
    axplot_Gmix(axs[1,j], ds, net_ebm, bsvec, fsig_best, jcx)


axplot_Gmix(axs[1,3], ds, net_ebm, bsvec, fsig_best0, jcx_long_tails, ' - with long tails')    
axplot_fsig_best(axs[0,3], fsig_best, lvec)
plt.savefig(ppath+'ds'+str(ds_opt)+'_KL_summary'+'.pdf')