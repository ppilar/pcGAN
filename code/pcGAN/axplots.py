# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt



    
#plot representation on ax
def axplot_rep(ax, ptrue_rep, cdata, jcz, crange_opt = ''):    
    crange = ptrue_rep.get_crange(jcz, crange_opt = crange_opt)
    zvec = np.linspace(crange[0], crange[1], 500)    
    pvec = ptrue_rep.get_pvec(torch.tensor(zvec), jcz)
    
    ax.plot(zvec, pvec.detach().cpu().numpy(), label='KDE')
    ax.set_xlim(crange)
    
    return crange
    
#plot histogram on ax
def axplot_hist(ax, cdata, hist_opts, crange=None):
    if type(hist_opts) == int:
        ax.hist(cdata, bins=20, density=True, range=crange);
    else:
        ax.hist(cdata, bins=20, density=True, range=crange, color=hist_opts[0], alpha=hist_opts[1]);
    
#plot histogram together with representation on ax      
def axplot_rep_hist(rep, z, ax, jcz=False, hist=False, hist_opts = -1, xmax = False, c15_bounds = False, crange_opt = ''): 
    
    crange = axplot_rep(ax, rep, z, jcz, crange_opt = crange_opt)
    if hist:
        axplot_hist(ax, z, hist_opts, crange)
    
    
        
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
    

#plot mb history on ax
def axplot_recent_mbs(ax, Nh, pfake_rep, jc, xvec, cmb_label=True):
    buf_ges = torch.zeros(xvec.shape[0])
    for j in range(Nh-1): #plot N-1 recent mbs
        buf = pfake_rep.history[j][jc,:].detach().cpu()
        buf_ges += buf/Nh
        alpha = 1/(Nh)*(j+1)
        ax.plot(xvec, buf, color='grey', alpha = alpha)  
        
    label = 'current_mb' if cmb_label else ''
    ax.plot(xvec, pfake_rep.history[Nh-1][jc,:].detach().cpu(), color = 'black', label=label)
    
    
    
#####################
##################### from dataset

#plot distribution of a given dataset constraint/performance metric on ax;
#optionally, together with KDE representation in case of constraints
def axplot_metric(ds, ax, mbuf, cbuf, crange, ptrue_rep, jind, jm, j, Nr, bins, vals0, mopt, plot, atitle, metric_name, bleg=False, crange_opt = ''):
    if plot:
        vals, bins, _ = ax.hist(mbuf.cpu().numpy(), bins = bins, range=crange, density=True)
        if mopt == 'c':
            # if type(ptrue_rep) == list:
            #     axplot_rep_hist(ptrue_rep[jind], cbuf, ax)
            # else:
            axplot_rep_hist(ptrue_rep, cbuf, ax, jcz=jind, c15_bounds=ds.cplot15_bounds, crange_opt = crange_opt)
            if bleg: ax.legend()
        if j%Nr == 0: ax.set_title(atitle[jm])
        ax.set_xlabel(metric_name, labelpad=-1)
        if type(vals0) == int:
            vals0 = vals
        ax.set_ylim([0,vals0.max()*(1.1)])
    else:
        vals, bins = np.histogram(mbuf.cpu().numpy(), bins=bins, range=crange, density=True)
        
    return vals, bins

#plot individual sample of dataset on given ax
def axplot_samples(ds, ax, sbuf, dscores=-1):
    plt.sca(ax)
    if ds.sample_ptype == plt.bar:
        ds.sample_ptype(ds.fftrange, sbuf, width=ds.bar_width)
    else:
        #ds.sample_ptype(ds.xvec, sbuf)
        ds.sample_ptype(sbuf)
        
    if type(dscores) != int:
        ax.set_title('D=%.3f'%(dscores))