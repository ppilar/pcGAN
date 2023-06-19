# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import generate_Ns

#plot ebm on ax
def plot_ebm_ax(ebm, z, ax, jcz=False, hist=False, hist_opts = -1, xmax = False, c15_bounds = False):
    conditional = type(jcz) != False
    
    off = torch.maximum(torch.abs(ebm.cmaxs[jcz]), torch.abs(ebm.cmins[jcz])).cpu()*0.05
    zvec = np.linspace(ebm.cmins[jcz].cpu()-off, ebm.cmaxs[jcz].cpu() + off, 500)
    if jcz == 15 and c15_bounds: #used to focus on relevant region of plot
        zvec = np.linspace(ebm.cmins[jcz].cpu()-off/3, ebm.cmaxs[jcz].cpu()/6 + off/6, 500)
        
    if conditional:
        clabels = jcz*torch.ones(500, device=ebm.device).int()
        pbuf = ebm((torch.tensor(zvec, device=ebm.device).unsqueeze(1).float(), clabels))
        mbuf = pbuf.max()
        pvec = np.exp((pbuf-mbuf).detach().cpu().numpy())
    else:
        pvec = np.exp(ebm(torch.tensor(zvec, device=device).unsqueeze(1).float()).detach().cpu().numpy())
        
        
    norm_ebm = np.trapz(pvec[:,0],zvec)  
    ax.plot(zvec, pvec/norm_ebm, label='EBM');
    
    
    if type(xmax) == bool:
        hist_xmax = ebm.cmaxs[jcz].item()
    else:
        hist_xmax = xmax
    
    if hist==True:
        if type(hist_opts) == int:
            ax.hist(z, bins=20, density=True);
        else:
            ax.hist(z, bins=20, density=True, color=hist_opts[0], alpha=hist_opts[1]);
                
    if type(xmax) == bool:
        ax.set_xlim((zvec[0], zvec[-1]))
    else:
        ax.set_xlim((zvec[0], xmax))

#plot cebm PDFs together with histograms of real/generated data
def plot_cebm(ebm, cdata, clabels, ppath=''):
    print('plot ebm')
    
    Nebm = cdata.shape[1]
    Nr = int(np.sqrt(Nebm))+1
    Nc = Nr-1 if Nr*(Nr-1) >= Nebm else Nr
    
    fig, axs = plt.subplots(Nr,np.maximum(Nc,2), figsize=(30,35))
    for jcx in range(Nebm):
        ax = axs[jcx//Nc, jcx%Nc]
        plot_ebm_ax(ebm, cdata[:,jcx].cpu().numpy(), ax, jcx, True)
        ax.set_xlabel(clabels[jcx])
    plt.savefig(ppath+'cebm_'+str(ebm.Nc)+'.pdf')
    plt.show()

#plot losses vs iterations
def plot_losses(losses, spath):
    i0 = 0
    imax = len(losses[0][i0::10])
    xbuf = 10*np.linspace(0,imax-1,imax) + i0
    plt.figure()
    plt.plot(xbuf, losses[0][i0::10],label='d0')
    plt.plot(xbuf, losses[1][i0::10],label='d2')
    plt.plot(xbuf, losses[2][i0::10],label='g0')
    plt.plot(xbuf, losses[3][i0::10],label='g2')
    ldmax = np.abs(losses[0]).max()
    plt.ylim([-ldmax,ldmax])
    plt.legend()
    plt.xlabel('iterations')
    plt.savefig(spath)
    plt.show()

     
#get range of frequencies
def get_fftrange(xvec):
    Nx = xvec.shape[0]
    lx = xvec[-1]
    fftrange = np.linspace(1,Nx,Nx)*2*np.pi/lx
    
    return fftrange
    
#plot hists of Fourier components
def plot_fftges(fftges):
    fig, axs = plt.subplots(5,5, figsize=(12,12))
    for j in range(25):
        axs[j//5,j%5].hist(fftges[:,j])
        
#plot intervals of constraint values vs iterations on ax
def axplot_metric_trajectory_intervals(ax, xbuf, cbuf, cKLs_real, ybounds = [1e-4, 1e8], label='mean'):
    cmean = np.nanmean(cbuf,1)
    cstd = np.nanstd(cbuf,1)
    cmin = np.nanmin(cbuf,1)
    cmax = np.nanmax(cbuf,1)
    ax.semilogy(xbuf, cmean, label=label)
    ax.fill_between(xbuf, cmin, cmax, alpha=0.5)
    ax.title.set_text('constraints - range')
    ax.set_xlabel('iterations')
    ax.set_ylim(ybounds)
    
    cKLs_real_mean = np.nanmean(cKLs_real)
    cKLs_real_max = np.nanmax(cKLs_real)
    cKLs_real_min = np.nanmin(cKLs_real)
    ax.axhline(y=cKLs_real_mean, c="black", linewidth=2, label='real')
    ax.axhline(y=cKLs_real_min, c="black", linestyle='--', linewidth=2)
    ax.axhline(y=cKLs_real_max, c="black", linestyle='--', linewidth=2)
    print('2')
    
#plot metrics vs iterations on ax
def axplot_metric_trajectories(ax, xbuf, pbuf, pKLs_real, plegend, linestyle='-', colors=-1):
    if type(linestyle) == str:
        linestyle = [linestyle for j in range(pbuf.shape[1])]
    
    for j in range(pbuf.shape[1]):
        if j != 0 and colors==-1:
            c = l[0].get_color()
            ax.semilogy(xbuf, pbuf[:,j], linestyle[j], color=c)
        else:
            l = ax.semilogy(xbuf, pbuf[:,j], linestyle[j])

    
    pKLs_real_mean = np.nanmean(pKLs_real)
    pKLs_real_max = np.nanmax(pKLs_real)
    pKLs_real_min = np.nanmin(pKLs_real)
    ax.axhline(y=pKLs_real_mean, c="black", linewidth=2)
    ax.axhline(y=pKLs_real_min, c="black", linestyle='--', linewidth=2)
    ax.axhline(y=pKLs_real_max, c="black", linestyle='--', linewidth=2)
    
    ax.legend(plegend)
    ax.title.set_text('performance metrics')
    ax.set_xlabel('iterations')    
    ax.set_ylim([1e-4,1e4])
    
    
#plot of metrics + constraint intervals vs iterations
def plot_metric_trajectories(constraints_ges, pmetrics_ges, KLs_real, clegend, plegend, eval_it, spath=''): 
    cKLs_real = KLs_real[0].cpu()
    pKLs_real = KLs_real[1].cpu()
    
    
    cbuf = np.array(constraints_ges)
    pbuf = np.array(pmetrics_ges)
    xbuf = eval_it*np.linspace(0,cbuf.shape[0]-1,cbuf.shape[0])
    
    
    fig, axs = plt.subplots(1,3, figsize=(24,4))
    axs[0].semilogy(xbuf, cbuf)
    if len(clegend) <= 10: axs[0].legend(clegend)
    axs[0].title.set_text('constraints')
    axs[0].set_xlabel('iterations')
    axs[0].set_ylim([1e-4,1e4])
    
    axplot_metric_trajectory_intervals(axs[1], xbuf, cbuf, cKLs_real)
    axs[1].legend()    
    axplot_metric_trajectories(axs[2], xbuf, pbuf, pKLs_real, plegend, colors=1)
    
    if spath != '':
        plt.savefig(spath+'metric_trajectories.pdf')
    plt.show()