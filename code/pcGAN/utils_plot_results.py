# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .axplots import axplot_metric_trajectory_intervals, axplot_metric_trajectories, axplot_samples


def plot_constraints_and_pmetrics(res, ds, ptrue_rep, ds_opt, pinds, ppath):
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
        
        axplot_metric_trajectory_intervals(axs[0], xbuf, cbuf, ptrue_rep.cKLs_real.cpu(), ybounds = [1e-4,1e8], label='jm'+str(j))
        axplot_metric_trajectories(axs[1], xbuf, pbuf[:,pinds], ptrue_rep.pKLs_hist_real[pinds].cpu(), np.array(ds.pmetric_names)[pinds], linestyle=mls)
    
    
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
        
    plt.savefig(ppath + 'metrics_summary.pdf', bbox_inches='tight')
    
    
def plot_sample_comparison(ds, gges, ds_opt, atitle, ppath):
    Nsp = 7
    Nm = len(gges)
    
    if ds_opt in [1,3]:
        fsize = (3.6*Nm+1, Nsp*3)
    else:
        fsize = (12,21)
    
    
    i0 = np.random.randint(19000)
    fig, axs = plt.subplots(Nsp, Nm+1, figsize=fsize)
    fig.subplots_adjust(wspace=0.3)
    for jd in range(Nm+1):
        if jd > 0:
            sbuf = gges[jd-1][i0:i0+Nsp]
        else:
            sbuf = ds.data[i0:i0+Nsp]
        for j in range(Nsp):
            axplot_samples(ds, axs[j,jd], sbuf[j].cpu())
            #ds.axplot_samples(axs[j,jd], sbuf[j].cpu())
            if j == 0:
                axs[j,jd].set_title(atitle[jd])
            if ds_opt == 2:
                axs[j,jd].get_xaxis().set_ticks([])
                axs[j,jd].get_yaxis().set_ticks([])
            
    plt.savefig(ppath + 'samples_summary.pdf', bbox_inches='tight')