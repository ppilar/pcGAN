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

from pcGAN.results import Results
from pcGAN.cebm import cEBM
from pcGAN.utils import *
from pcGAN.plots import *
from pcGAN.datasets import *
from pcGAN.utils_summary_plots import initialize_cebm_ds

#%%
rand_init, s = init_random_seeds(s=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds_opt = 1
ds, ebms, res, _, _, _, cinds_selection, pinds, ppath = initialize_cebm_ds(ds_opt) #load trained cebm + corresponding dataset

#%% plot constraints and performance metrics vs iterations

plt.rcParams.update({'font.size': 14})
mls = ['-','--',':']
als = ['--','-','-.',':']

fig, axs = plt.subplots(2,1, figsize=(10,10))
fig.tight_layout(h_pad=3)
aj = [1,0,3]
for j in aj:
    cbuf = np.array(res.constraints_ges[j])
    pbuf = np.array(res.pmetrics_ges[j])
    xbuf = res.eval_it*np.linspace(0,cbuf.shape[0]-1,cbuf.shape[0])    
    
    axplot_metric_trajectory_intervals(axs[0], xbuf, cbuf, ebms.cKLs_real.cpu(), ybounds = [1e-4,1e8], label='jm'+str(j))
    axplot_metric_trajectories(axs[1], xbuf, pbuf[:,pinds], ebms.pKLs_hist_real[pinds].cpu(), np.array(ds.pmetric_names)[pinds], linestyle=mls)


if ds_opt == 1:
    axs[0].set_ylim(1e-3,1e4)
    axs[1].set_ylim(3e-2,1e2)
elif ds_opt == 2:
    axs[0].set_ylim(1e-2,3e0)
    axs[1].set_ylim(1e-3,2e1)    
elif ds_opt == 3:
    axs[0].set_ylim(1e0,1e1)
    axs[1].set_ylim(1e-3,1e1)

handles, labels = axs[0].get_legend_handles_labels()
new_handles = [handles[0], handles[2], handles[4], handles[5]]
new_labels = ['pcGAN', 'WGAN', 'Wu et al.', 'real data']
order = [1,0,2,3]
axs[0].legend([new_handles[i] for i in order], [new_labels[i] for i in order],  loc='upper right')

leg = axs[1].legend(np.array(ds.pmetric_names)[pinds], loc='upper right')
for j in range(len(pinds)):
    leg.legendHandles[j].set_color('k')
    
plt.savefig(ppath + 'metrics_summary.pdf')

#%% generate data for the different GANs
gges = []
for jm in [0,1,3]:
    gnet = res.gnet_ges[jm][0]
    gges.append(generate_Ns(gnet, 20000, device, gnet.latent_dim).squeeze())


#%% plot comparison of constraint fulfillment
atitle = ['real','WGAN','pcGAN','Wu et al.']
ds.cplot15_bounds = True
ds.cinds_selection = cinds_selection
ds.plot_constraints(gges, ebms, spath=ppath+'constraint_comparison', atitle=atitle, all_constraints=False);

#%% plot real and generated samples
if ds_opt == 1:
    fsize = (14,21)
else:
    fsize = (12,21)

Nsp = 7
i0 = np.random.randint(19000)
fig, axs = plt.subplots(Nsp, 4, figsize=fsize)
for jd in range(4):
    if jd > 0:
        sbuf = gges[jd-1][i0:i0+Nsp]
    else:
        sbuf = ds.data[i0:i0+Nsp]
    for j in range(Nsp):
        ds.axplot_samples(axs[j,jd], sbuf[j].cpu())
        if j == 0:
            axs[j,jd].set_title(atitle[jd])
        if ds_opt == 2:
            axs[j,jd].get_xaxis().set_ticks([])
            axs[j,jd].get_yaxis().set_ticks([])
        
plt.savefig(ppath + 'samples_summary.pdf')

