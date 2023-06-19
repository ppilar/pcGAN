# -*- coding: utf-8 -*-
import pickle
import torch
import numpy as np
from .plots import plot_ebm_ax

#initialize parameters and load results for ds_opt
def initialize_cebm_ds(ds_opt):
    Ns = 100000
    use_cvalue_input = False
    if ds_opt == 1:
        path0 = '../results/ds1_Nd_1_bs_256/wave_forms/'
        ajcx = [0, 15, 30]
        jcx_long_tails = 15
        KL_ylim = [1e-2, 5e1]
        cinds_selection = [1, 15, 50]
        pinds = [0,4,5]
    if ds_opt == 2:
        path0 = '../results/ds2_Nd_1_bs_256/Tmaps/'
        ajcx = [0, 15, 30]
        jcx_long_tails = 0
        KL_ylim = [1e-2, 5e1]
        cinds_selection = [1, 15, 30]
        pinds = [1,4,7]
    if ds_opt == 3:
        path0 = '../results/ds3_Nd_1_bs_256/IceCube/'
        ajcx = [0,1]
        jcx_long_tails = 0
        KL_ylim = [1e-2, 5e1]  
        cinds_selection = [0,1]
        pinds = [1,4,5]
        
    ppath = path0 + 'plots/'
    fpath = path0 + 'files/'
    fname = 'ds' + str(ds_opt) + '_results.pk'
    with open(fpath + fname, 'rb') as f:
        ds, net_ebm, res = pickle.load(f)
    
    return ds, net_ebm, res, ajcx, jcx_long_tails, KL_ylim, cinds_selection, pinds, ppath
        
#calculate PDF from superposition fo Gaussians
def superposition_of_gaussians(x, points, sigma=0.5):
    y = np.exp(-(x[:, np.newaxis]-points)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    return y.mean(axis=1)

#sample a minibatch
def sample_mb(ds, bs=64):
    perm = torch.randperm(ds.data.size(0))
    idx = perm[:bs]
    return ds.data[idx], ds.constraints[idx]

#plot KL curve on ax
def axplot_KLdiv(ax, fsig_vec, KLs, title, lvec, ylim=[1e-2, 5e1], yl_opt = False):
    ax.loglog(fsig_vec, KLs.mean(-1).T)
    ax.legend(lvec)
    ax.set_xlabel('$f_{\sigma}$')
    if yl_opt:
        ax.set_ylabel('KL divergence')
    ax.set_title(title)
    ax.set_ylim(ylim)

#plot Gaussian mixture on ax
def axplot_Gmix(ax, ds, net_ebm, bsvec, fsig_best, jcx, xl_str = '', xmax=False):
    cmin = ds.constraints[:,jcx].min().cpu()
    cmax = ds.constraints[:,jcx].max().cpu()
    x = np.linspace(cmin,cmax,100)
    
    for jbs, bs in enumerate(bsvec[:]):        
        _, cmb = sample_mb(ds, bs)
        cmb = cmb.cpu().numpy()
        y = superposition_of_gaussians(x, cmb[:,jcx], net_ebm.cstds[jcx].cpu().numpy()/fsig_best[jcx,jbs])
        ax.plot(x,y,'--',label='bs='+str(bs))
    
    if type(xmax) == bool:
        plot_ebm_ax(net_ebm, ds.constraints[:,jcx].cpu(), ax, jcz=jcx, hist=True, hist_opts=('grey',0.3))        
    else:
        plot_ebm_ax(net_ebm, ds.constraints[:,jcx].cpu(), ax, jcz=jcx, hist=True, hist_opts=('grey',0.3), xmax = xmax)

    ax.get_lines()[-1].set_color("black")
    ax.legend()
    ax.set_xlabel(ds.constraint_names[jcx] + xl_str)
        
#plot values fsig_best on ax
def axplot_fsig_best(ax, fsig_best, lvec):
    ax.plot(fsig_best)
    ax.legend(lvec)
    ax.set_xlabel('i')
    ax.set_ylabel('$f_{\sigma}^*$');
    
        