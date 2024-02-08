# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import generate_Ns, constraint_error_hist
from .axplots import *


#plot cebm PDFs together with histograms of real/generated data
def plot_rep(rep, cdata, clabels, ppath=''):
    print('plot rep')
    
    Ns = cdata.shape[1]
    Nr = int(np.sqrt(Ns))+1
    Nc = Nr-1 if Nr*(Nr-1) >= Ns else Nr
    
    fig, axs = plt.subplots(Nr,np.maximum(Nc,2), figsize=(30,35))
    for jcx in range(Ns):
        ax = axs[jcx//Nc, jcx%Nc]
        axplot_rep_hist(rep, cdata[:,jcx].cpu().numpy(), ax, jcx, True)
        ax.set_xlabel(clabels[jcx])
    plt.savefig(ppath+'rep_'+str(rep.Nc)+'.pdf', bbox_inches='tight')
    plt.show()
    return fig

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
    plt.savefig(spath, bbox_inches='tight')
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
        

    
    
#plot of metrics + constraint intervals vs iterations
def plot_metric_trajectories(constraints_ges, pmetrics_ges, KLs_real, clegend, plegend, eval_it, spath=''): 
    cKLs_real = KLs_real[0]#.cpu()
    pKLs_real = KLs_real[1]#.cpu()
    
    
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
        plt.savefig(spath+'metric_trajectories.pdf', bbox_inches='tight')
    plt.show()
    
    
    
    
    
    
####################
#################### from dataset


#plot distributions of constraints or performance metrics
def plot_metrics(ds, gdata, metrics, f_calculate_metrics, metric_names, spath, match_opt, mopt='p', ptrue_rep=-1, plot=True, inds_plot = -1, atitle=-1, crange_opt = ''):
    Nmodel = len(gdata) + 1
    Ns = gdata[0].shape[0]
    
    Nmetric = metrics.shape[1]
    if type(inds_plot) == int: inds_plot = range(Nmetric)
    Nplot = len(inds_plot)
    
    ametrics = []
    ametrics.append(metrics)
    for j in range(0,Nmodel-1):
        ametrics.append(f_calculate_metrics(gdata[j]))
    
    metrics_RMSEs = []
    metrics_KLs = []
    
    fig = -1
    if plot:
        if Nplot > 20:
            Nr = 20
            Nc = (Nplot//Nr + 1)*Nmodel
            fig, axs = plt.subplots(Nr, Nc, figsize=(2.25*Nmodel*Nc,4*Nr))
        else:
            Nr = Nplot
            fig, axs = plt.subplots(Nplot, Nmodel, figsize=(4.5*Nmodel,4*Nplot))
            plt.subplots_adjust(hspace=0.25)
    else:
        axs = -1
        Nr = -1
           
    if type(atitle) == int:
        atitle = ['true']
        for j in range(Nmodel-1):
            atitle.append('fake')
            
    for j, jind in enumerate(inds_plot):            
        if mopt == 'c':
            cbuf = ds.constraints[:,jind].cpu().numpy()
            crange = ptrue_rep.get_crange(jind, crange_opt = crange_opt)
            
        else:
            cbuf = -1
            if metric_names[jind] == 'E' and ds.ds_name == 'IceCube':
                crange = (0, 3000)
            elif metric_names[jind] == 'E' and ds.ds_name == 'wave_forms':
                crange = (0, 15000)
            else:
                crange=None
        
        bins = 25
        vals0 = -1
        
        
        for jm in range(Nmodel):
            mbuf = ametrics[jm]
            if plot:
                ax = axs[jm] if Nmetric == 1 else axs[j%Nr,(j//Nr)*Nmodel + jm]
            else:
                ax = -1
            
            bleg = j%Nr == 0 and (j//Nr)*Nmodel + jm == 0
            vals1, binbuf = axplot_metric(ds, ax, mbuf[:,jind], cbuf, crange, ptrue_rep, jind, jm, j, Nr, bins, vals0, mopt, plot, atitle, metric_names[jind], bleg, crange_opt = crange_opt)
            if jm == 0:
                bins = binbuf
                vals0 = vals1
                            
        #store values
        metrics_RMSEs.append(np.sqrt(np.mean((vals0-vals1)**2)))
        metrics_KLs.append(constraint_error_hist(vals0, vals1, bins, match_opt))
        
    if plot:
        if spath != '':
            plt.savefig(spath, bbox_inches='tight')
        plt.show()
    
    return fig, metrics_KLs



        
#plot distributions of performance metrics
def plot_pmetrics(ds, gdata, spath, match_opt, plot=True, atitle=-1, crange_opt = ''):
    if type(gdata) != list: gdata = [gdata]
    return plot_metrics(ds, gdata, ds.pmetrics, ds.calculate_performance_metrics, ds.pmetric_names, spath+'pmetrics.pdf', match_opt, plot=plot, atitle=atitle, crange_opt=crange_opt)
    
#plot distributions of constraints
def plot_constraints(ds, gdata, ptrue_rep, spath, match_opt, plot=True, atitle=-1, all_constraints = True, crange_opt = ''):
    if type(gdata) != list: gdata = [gdata]
    if type(ds.cinds_selection) != int and all_constraints==False:
        if np.max(ds.cinds_selection) < ds.Nebm:
            return plot_metrics(ds, gdata, ds.constraints[:,:ds.Nebm], ds.calculate_constraints, ds.constraint_names, spath+'constraints_selection.pdf', match_opt, 'c', ptrue_rep, plot=plot, inds_plot=ds.cinds_selection, atitle=atitle, crange_opt = crange_opt)
    if all_constraints:
        return plot_metrics(ds, gdata, ds.constraints[:,:ds.Nebm], ds.calculate_constraints, ds.constraint_names, spath+'constraints.pdf', match_opt, 'c', ptrue_rep, plot=plot, atitle=atitle, crange_opt = crange_opt)
    
    
#plot a summary of constraints and performance metrics on generated data
def plot_summary(ds, gdata, spath):
    
    cmeans = ds.constraints.mean(0).cpu().numpy()
    pmeans = ds.pmetrics.mean(0).cpu().numpy()
    
    pmetrics_gen = ds.calculate_performance_metrics(gdata)
    constraints_gen = ds.calculate_constraints(gdata)
    pmeans_gen = pmetrics_gen.mean(0).cpu().numpy()
    cmeans_gen = constraints_gen.mean(0).cpu().numpy()
    
    fig, axs = plt.subplots(2,2, figsize=(12,5))
    axs[0,0].bar(np.arange(cmeans.shape[0]),cmeans)
    axs[0,1].bar(np.arange(pmeans.shape[0]), pmeans)
    
    axs[0,0].title.set_text('constraints')
    axs[0,1].title.set_text('pmetrics')
    
    axs[1,0].bar(np.arange(cmeans_gen.shape[0]),cmeans_gen)
    axs[1,1].bar(np.arange(pmeans_gen.shape[0]), pmeans_gen)
    
    if spath != '':
        plt.savefig(spath + '.pdf', bbox_inches='tight')
    plt.show()
    
#plot a number of dataset samples
def plot_samples(ds, samples, spath=''): 
    if type(samples) == tuple:
        dscores = samples[1]
        samples = samples[0]
    else:
        dscores = -1
    sbuf = samples.detach().cpu().numpy()
    
    #### 1x6 plots
    fig, axs = plt.subplots(1, 6, figsize=ds.s6_figsize)
    for j in range(6):
        axplot_samples(ds, axs[j], sbuf[j], dscores[j])
    
    if spath != '':
        plt.savefig(spath+'_' + ds.ds_name + '_6.pdf', bbox_inches='tight')
    plt.show()
    
    #### 4x4 plots
    
    fig, axs = plt.subplots(4,4, figsize=ds.s16_figsize)
    for j in range(16):
        axplot_samples(ds, axs[j//4, j%4], sbuf[j], dscores[j])
    fig.subplots_adjust(hspace=0.5)
    
    
    if spath!= '':
        plt.savefig(spath + '_' + ds.ds_name + '_16.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


#plot ways of weighting minibatches
def plot_mb_weighting(ptrue_rep, pfake_rep, jc, path = ''):
    Nh = pfake_rep.Nh
    xvec = pfake_rep.xvals[0,jc,:].detach().cpu()
    
    plt.figure()
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    axplot_recent_mbs(axs[0], Nh, pfake_rep, jc, xvec)
    axs[0].plot(xvec, pfake_rep.weighted_average()[jc].cpu(), label='with history', color='green')
    axplot_cebm(axs[0], ptrue_rep, -1, jc)
    axs[0].legend()
    axs[0].set_title('weighted average with N=%i' % (pfake_rep.Nh))
    
    axplot_recent_mbs(axs[1], Nh, pfake_rep, jc, xvec)
    axs[1].plot(xvec, pfake_rep.exponential_decay()[jc].cpu(), label='with history', color='green')
    axplot_cebm(axs[1], ptrue_rep, -1, jc)
    axs[1].legend()
    axs[1].set_title('exponential decay with f= %.2f' % (pfake_rep.fforget))
    
    plt.savefig(path+'batch_history_'+str(jc)+'.pdf', bbox_inches='tight')
    
    

#plot how well the various constraints are fulfilled
def plot_constraint_fulfillment(closses, closses0, counts, spath=''):
    N = closses.shape[0]
    def f(x):
        return np.log10(np.abs(x))#*np.sign(x)
    
    fig, axs = plt.subplots(3,1,figsize=(7,14))    
    axs[0].bar(range(N), f(closses))
    axs[0].set_title('log(absolute values)')
    axs[0].set_ylim(-3,1)
    #axs[0].set_ylim(0,3)
    axs[1].bar(range(N), f(closses0))
    axs[1].set_title(r'log($\Delta$ with real data value)')
    axs[1].set_ylim(-3,1)
    #axs[1].set_ylim(-0.5,2.5)
    axs[2].bar(range(N), counts/counts.sum())
    axs[2].set_title('constraint sampling counts')
    axs[2].set_ylim(0,0.06)
    axs[2].set_xlabel('constraints')
    
    if spath!= '':
        plt.savefig(spath+'constraint_fulfillment.pdf', bbox_inches='tight')    
    plt.show()
    
    return fig
    