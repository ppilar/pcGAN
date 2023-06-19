# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import torch
import pickle
from quantimpy import minkowski as mk

from .plots import plot_ebm_ax
from .Nets import *

import matplotlib.pyplot as plt

class dataset():
    def __init__(self, Ns, device, pars):
        self.Ns = Ns
        self.device = device
        self.init_cp_names()
        self.data = self.generate_data(Ns)
        self.constraints = self.calculate_constraints(self.data)
        self.Nc = len(self.constraint_names)
        self.Nebm = self.Nc
        self.cmeans, self.cstds = self.calculate_constraint_stats(self.constraints)
        self.dcov = self.calculate_covariance(self.data)
        self.pmetrics = self.calculate_performance_metrics(self.data)
        self.Np = self.pmetrics.shape[1]
        self.latent_dim = pars[0]
        self.use_cvalue_input = pars[1]
        
        
        self.lr = 2e-4
        self.fsched = 0.2
        
        self.cut_tails = True
        
        self.fcov = 1
        self.kl_opt = 'G'
        
        self.s6_figsize = (20,3)
        self.s16_figsize = (12,12)
        self.bar_width = 0.1
        
        self.cinds_selection=-1
        self.Nit_ebm = 20000
        
        
        self.cplot15_bounds = False #use specific bounds for plot
        
    #generate Ns samples
    def generate_data(self, Ns):
        raise NotImplementedError("Function not implemented!")
        
    #calculate constraints for all samples in data
    def calculate_constraint(self, data):
        raise NotImplementedError("Function not implemented!")
        
    #calculate covariance of samples; used for approach from Wu et al.
    def calculate_covariance(self, data):
        if data.ndim > 2:
            data = data.flatten(1,2)
        return torch.cov(data.squeeze().T) 
        
    #define list fo constraints and performance metrics according to their names
    def init_cp_names(self):
        raise NotImplementedError("Function not implemented!")
        
    #initialize networks, optimizers and schedulers
    def initialize_models(self):
        gnet, dnet = self.get_GAN_nets()
        gopt, dopt = self.get_optimizers(gnet, dnet)
        gsched, dsched = self.get_schedulers(gnet, dnet, gopt, dopt)
        return gnet, dnet, gopt, dopt, gsched, dsched
        
    #initialize generator and discriminator
    def get_GAN_nets(self, pars):
        raise NotImplementedError("Function not implemented!")
        
    #initialize optimizers
    def get_optimizers(self, gnet, dnet):
        gopt = torch.optim.Adam(gnet.parameters(), lr=self.lr, betas = (0, 0.9))
        dopt = torch.optim.Adam(dnet.parameters(), lr=self.lr, betas = (0, 0.9))
        return gopt, dopt
    
    #initialize learning rate schedulers
    def get_schedulers(self, gnet, dnet, gopt, dopt):
        gsched = torch.optim.lr_scheduler.ExponentialLR(gopt, self.fsched)
        dsched = torch.optim.lr_scheduler.ExponentialLR(dopt, self.fsched)
        return gsched, dsched
    
    #allow for data augmentation; if no augmentation, return unaugmented batch
    def augment_batch(self, batch, dim_offset = 0):
        return batch
    
    #calculate constraint statistics
    def calculate_constraint_stats(self, constraints):
        cmeans = constraints.mean(0)
        cstds = constraints.std(0)
        return cmeans, cstds
    
    #calculate constraints or performance metrics on data
    def calculate_metrics(self, data, cpnames):
        raise NotImplementedError("Function not implemented!")
    
    #calculate constraints on data
    def calculate_constraints(self, data):
        return self.calculate_metrics(data, self.constraint_names)
    
    #calculate performance metrics on data
    def calculate_performance_metrics(self, data):
        return self.calculate_metrics(data, self.pmetric_names)
    
    #plot a summary of constraints and performance metrics on generated data
    def plot_summary(self, gdata, spath):
        
        cmeans = self.constraints.mean(0).cpu().numpy()
        pmeans = self.pmetrics.mean(0).cpu().numpy()
        
        pmetrics_gen = self.calculate_performance_metrics(gdata)
        constraints_gen = self.calculate_constraints(gdata)
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
            plt.savefig(spath + '.pdf')
        plt.show()
        
    #plot individual sample of dataset on given ax
    def axplot_samples(self, ax, sbuf, dscores=-1):
        plt.sca(ax)
        if self.sample_ptype == plt.bar:
            self.sample_ptype(self.fftrange, sbuf, width=self.bar_width)
        else:
            #self.sample_ptype(self.xvec, sbuf)
            self.sample_ptype(sbuf)
            
        if type(dscores) != int:
            ax.set_title('D=%.3f'%(dscores))
    
    #plot a number of dataset samples
    def plot_samples(self, samples, spath=''):
        if type(samples) == tuple:
            dscores = samples[1]
            samples = samples[0]
        else:
            dscores = -1
        sbuf = samples.detach().cpu().numpy()
        
        #### 1x6 plots
        fig, axs = plt.subplots(1, 6, figsize=self.s6_figsize)
        for j in range(6):
            self.axplot_samples(axs[j], sbuf[j], dscores[j])
        
        if spath != '':
            plt.savefig(spath+'_' + self.ds_name + '_6.pdf')
        plt.show()
        
        #### 4x4 plots
        
        fig, axs = plt.subplots(4,4, figsize=self.s16_figsize)
        for j in range(16):
            self.axplot_samples(axs[j//4, j%4], sbuf[j], dscores[j])
        fig.subplots_adjust(hspace=0.5)
        
        
        if spath!= '':
            plt.savefig(spath + '_' + self.ds_name + '_16.pdf')
        plt.show()
        
    #plot distribution of a given dataset constraint/performance metric on ax;
    #optionally, together with cEBM representation in case of constraints
    def axplot_metric(self, ax, mbuf, cbuf, crange, ebms, jind, jm, j, Nr, bins, vals0, mopt, plot, atitle, metric_name, bleg=False):
        if plot:
            vals, bins, _ = ax.hist(mbuf.cpu().numpy(), bins = bins, range=crange, density=True)
            if mopt == 'c':
                if type(ebms) == list:
                    plot_ebm_ax(ebms[jind], cbuf, ax)
                else:
                    plot_ebm_ax(ebms, cbuf, ax, jcz=jind, c15_bounds=self.cplot15_bounds)
                    if bleg: ax.legend()
            if j%Nr == 0: ax.set_title(atitle[jm])
            ax.set_xlabel(metric_name, labelpad=-1)
            if type(vals0) == int:
                vals0 = vals
            ax.set_ylim([0,vals0.max()*(1.1)])
        else:
            vals, bins = np.histogram(mbuf.cpu().numpy(), bins=bins, range=crange, density=True)
            
        return vals, bins
    
    #plot distributions of constraints or performance metrics
    def plot_metrics(self, gdata, metrics, f_calculate_metrics, metric_names, spath, mopt='p', ebms=-1, plot=True, inds_plot = -1, atitle=-1):
        Nmodel = len(gdata) + 1
        Ns = gdata[0].shape[0]
        
        Nmetric = metrics.shape[1]
        if type(inds_plot) == int: inds_plot = range(Nmetric)
        Nplot = len(inds_plot)
        
        ametrics = []
        ametrics.append(metrics)
        for j in range(0,Nmodel-1):
            ametrics.append(f_calculate_metrics(gdata[j]))
        #metrics_gen = f_calculate_metrics(gdata)
        
        metrics_RMSEs = []
        metrics_KLs = []
        
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
                cbuf = self.constraints[:,jind].cpu().numpy()
                #crange=(0,ebms.cmaxs_eff[jind].item())
                crange=(ebms.cmins[jind].item(), ebms.cmaxs[jind].item())
                if jind == 15:                    
                    crange=(ebms.cmins[jind].item(), ebms.cmaxs[jind].item()/6)
            else:
                cbuf = -1
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
                vals1, binbuf = self.axplot_metric(ax, mbuf[:,jind], cbuf, crange, ebms, jind, jm, j, Nr, bins, vals0, mopt, plot, atitle, metric_names[jind], bleg)
                if jm == 0:
                    bins = binbuf
                    vals0 = vals1
                                
            #store values
            metrics_RMSEs.append(np.sqrt(np.mean((vals0-vals1)**2)))
            metrics_KLs.append(KL_hist(vals0, vals1, bins))
            
        if plot:
            if spath != '':
                plt.savefig(spath)
            plt.show()
        
        return metrics_KLs
        
        
    #plot distributions of performance metrics
    def plot_pmetrics(self, gdata, spath, plot=True, atitle=-1):
        if type(gdata) != list: gdata = [gdata]
        return self.plot_metrics(gdata, self.pmetrics, self.calculate_performance_metrics, self.pmetric_names, spath+'pmetrics.pdf', plot=plot, atitle=atitle)
        
    #plot distributions of constraints
    def plot_constraints(self, gdata, ebms, spath, plot=True, atitle=-1, all_constraints = True):
        if type(gdata) != list: gdata = [gdata]
        if type(self.cinds_selection) != int:
            if np.max(self.cinds_selection) < self.Nebm:
                self.plot_metrics(gdata, self.constraints[:,:self.Nebm], self.calculate_constraints, self.constraint_names, spath+'constraints_selection.pdf', 'c', ebms, plot=plot, inds_plot=self.cinds_selection, atitle=atitle)
        if all_constraints:
            return self.plot_metrics(gdata, self.constraints[:,:self.Nebm], self.calculate_constraints, self.constraint_names, spath+'constraints.pdf', 'c', ebms, plot=plot, atitle=atitle)
        
#calculate KL divergence between two histograms
def KL_hist(vals0, vals1, bins):
    lbin = np.diff(bins)
    inz = np.where(vals0 > 0)[0]

    KL = np.sum(vals0[inz]*np.log(vals0[inz]/(vals1[inz]+1e-9))*lbin[inz])
    return KL
       
    
###################################
###################################
    
#synthetic example: dataset of waveforms, i.e. superpositions of 2 sine waves
class wave_forms(dataset):
    def __init__(self, Ns = 1000, device='cpu', pars=()):
        super().__init__(Ns, device, pars)
        self.clamp = 5e-3 
        self.lr = 2e-4
        self.fsched = 0.5
        
        #prefactors for added loss terms
        self.fKL = 30
        self.fKLG = 50/6
        self.fm = 1000*3/2
        
        
        self.sample_ptype = plt.plot
        self.ds_name = 'wave_forms'
        self.cinds_selection = [0,1,2,5,10,15,30,50,75,100]
        
        self.s16_figsize = (12,13.5)
        self.plot_fft = False
        
        
        
    def init_cp_names(self):
        self.constraint_names = ['ps [' + str(j) + ']' for j in range(101)]
        self.pmetric_names = ['mean','mean(abs)','min','max','max-min','E']
            
        
        
    def generate_data(self, Ns):
        Nf = 2 #also change output layer in Net!
        lx=20
        Nx=200        
        nf=0
        
        xvec = np.linspace(0,lx,Nx)
        
        fs_ges = np.abs(np.random.normal(1,1,Nf*Ns)).reshape(Ns,Nf)
        #fs_ges = np.abs(np.random.normal(0.3,0.5,Nf*Ns)).reshape(Ns,Nf)
        #fs_ges = np.abs(np.random.lognormal(0.2,0.3,Nf*Ns)).reshape(Ns,Nf)
        #fs_ges = np.abs(np.random.uniform(0,5,Nf*Ns)).reshape(Ns,Nf)
        fs_ges = np.expand_dims(fs_ges,2)

        wvec_ges = np.sum(np.sin(np.kron(fs_ges, xvec)), 1)
        if nf > 0:
            nvec_ges = np.random.normal(0,nf, wvec_ges.size).reshape(wvec_ges.shape)
            wvec_ges += nvec_ges
            
        #optional: attenuate signals on both sides to obtain curves that look more similar to detector signals
        # att1 = 1/(1+np.exp(-(xvec - 5)*1.5))
        # att2 = 1 - 1/(1+np.exp(-(xvec - 15)*1.5))
        # wvec_ges *= att1
        # wvec_ges *= att2
                      
        
        self.xvec = torch.tensor(xvec)
        return torch.tensor(wvec_ges/Nf, device=self.device).float()
    
    def calculate_metrics(self, data, mlist):
        Ns = data.shape[0]
        Nm = len(mlist)
        
        
        fftges, fftstats = get_fft(data)
        metrics = torch.zeros(Ns, Nm).to(self.device)
        for jm, m in enumerate(mlist):
            if m == 'mean':
                metrics[:,jm] = data.mean(1)
            if m =='mean(abs)':
                metrics[:,jm] = torch.abs(data).mean(1)
            if m == 'min':
                metrics[:,jm] = data.min(1)[0]
            if m == 'max':
                metrics[:,jm] = data.max(1)[0]
            if m == 'max-min':
                metrics[:,jm] = data.max(1)[0] - data.min(1)[0]
            if m[:2] == 'ps':
                i = int(m[4:-1])
                metrics[:,jm] = torch.abs(fftges[:,i])
            if m == 'E':
                metrics[:,jm] = (torch.abs(fftges)**2).sum(1)
        
        return metrics
    
    
    def get_GAN_nets(self, nopt='big'):
        gnet = Generator_1D_large(ds=1, nopt = nopt, latent_dim = self.latent_dim, device=self.device).to(self.device)
        dnet = Discriminator_1D_large(ds=1, nopt = nopt, use_fft_input = self.use_cvalue_input).to(self.device)
        gnet.xvec = self.xvec
        return gnet, dnet
    
    def plot_samples(self, samples, spath=''):
        self.sample_ptype = plt.plot
        super().plot_samples(samples, spath + '_t')
        
        if self.plot_fft == True:
            if type(samples) == tuple:
                dscores = samples[1]
                samples=samples[0]
            
            Nfpr = 101#30
            self.sample_ptype = plt.bar
            fft_samples, _ = get_fft(samples, 'torch')
            self.fftrange = get_fftrange(self.xvec)[:Nfpr]
            super().plot_samples(torch.abs(fft_samples)[:,:Nfpr], spath + '_ps')
        

    
##################################
##################################

#IceCube-Gen2: radio-detector signals
class IceCube_wave_forms(wave_forms):
    def __init__(self, device='cpu',pars=()):
        super().__init__(-1, device, pars)        
        self.ds_name = 'IceCube'
        self.clamp = 1e-1
        
        self.fKL = 0.1
        self.cinds_selection = [0,3,5,8,30,50,70]
        self.cut_tails = False
        
        
    def init_cp_names(self):
        #self.constraint_names = ['ps [' + str(j) + ']' for j in range(101)]
        self.constraint_names = ['max','min']
        #self.constraint_names = ['ps [' + str(j) + ']' for j in range(1,20)]
        self.pmetric_names = ['mean','mean(abs)','min','max','max-min','E','ps [0]']
    
    
    
    def calculate_metrics(self, data, mlist):
        Ns = data.shape[0]
        Nm = len(mlist)
        
        
        fftges, fftstats = get_fft(data)
        metrics = torch.zeros(Ns, Nm).to(self.device)
        for jm, m in enumerate(mlist):
            if m == 'mean':
                metrics[:,jm] = data.mean(1)
            if m =='mean(abs)':
                metrics[:,jm] = torch.abs(data).mean(1)
            if m == 'min':
                metrics[:,jm] = data.min(1)[0]
            if m == 'max':
                metrics[:,jm] = data.max(1)[0]
            if m == 'max-min':
                metrics[:,jm] = data.max(1)[0] - data.min(1)[0]
            if m[:2] == 'ps':
                i = int(m[4:-1])
                metrics[:,jm] = torch.log(torch.abs(fftges[:,i]) + 1e-12)
            if m == 'E':
                metrics[:,jm] = (torch.abs(fftges)**2).sum(1)
        
        return metrics
        
    def generate_data(self, Ns):
        dpath = '../data/'
        fname = 'IceCube.npy'
        
        dbuf = np.load(dpath+fname)#*2000
        
        dbuf = dbuf/(np.abs(dbuf).max(1))[np.newaxis].T
        ibuf = np.random.choice(np.arange(dbuf.shape[0]), 50000)
        Cdata = torch.tensor(dbuf[ibuf], device = self.device).float()
        self.xvec = torch.linspace(0,200,200)
        self.Ns = Cdata.shape[0]
        return Cdata
    
    def get_GAN_nets(self, nopt='big'):
        gnet = Generator_IceCube_v0(latent_dim = self.latent_dim).to(self.device)
        dnet = Discriminator_IceCube_v0().to(self.device)
        gnet.xvec = self.xvec
        return gnet, dnet

    
    
#calculate real discrete Fourier transform
def get_fft(wf, backend='torch'):
    if backend == 'torch':
        fftges = torch.fft.rfft(wf, axis=1)
        fftmean = torch.mean(torch.abs(fftges),0)
        fftstd = torch.std(torch.abs(fftges),0)
    elif backend == 'numpy':
        fftges = np.fft.rfft(wf, axis=1)
        fftmean = np.mean(np.abs(fftges),0)
        fftstd = np.std(np.abs(fftges),0)
        
    return fftges, (fftmean, fftstd)

#get range of frequencies
def get_fftrange(xvec):
    Nx = xvec.shape[0]
    lx = xvec[-1]
    fftrange = np.linspace(1,Nx,Nx)*2*np.pi/lx    
    return fftrange

    
###################################
###################################
    
#Temperature maps from the CAMELS project
class CAMELS(dataset):
    def __init__(self, device='cpu', pars=()):
        
        self.set_power_spectrum_pars()
        super().__init__(-1, device, pars)
        self.clamp = 2e-2
        
        
        self.lr = 1e-4
        
        #prefactors for added loss terms
        self.fKL = 0.01
        self.fKLG = 0.0025 
        self.fm = 0.5
        self.fcov = 0.001
        
        self.ds_name = 'Tmaps'
        self.sample_ptype = plt.imshow
        
        self.s6_figsize = (9,6)
        self.s16_figsize = (12,12)
        
        
        self.cinds_selection = [0,1,2,5,10,20,30]
        
        
    def generate_data(self, Ns, jmax = 1):
        dpath = '../data/'
        fname = 'Tmaps.pickle'
        with open(dpath + fname,'rb') as f:
            Cdata = torch.tensor(pickle.load(f), device=self.device)

        self.Ns = Cdata.shape[0]
        
        return Cdata
    
    def init_cp_names(self):
        self.constraint_names = ['ps [' + str(j) + ']' for j in range(32)]
        #athr = [-0.5, 0.0, 0.5]
        athr = [0.8, 0.9, 0.95]
        self.pmetric_names = ['Mk 0 ('+str(a)+')' for a in athr ]
        self.pmetric_names.extend(['Mk 1 ('+str(a)+')' for a in athr ])
        self.pmetric_names.extend(['Mk 2 ('+str(a)+')' for a in athr ])
        #self.pmetric_names = ['Tpdf']
        
    def calculate_metrics(self, data, mlist):
        Ns = data.shape[0]
        Nm = len(mlist)
        
        
        #fftges, fftstats = get_fft(data)
        power_spectrum = self.calculate_power_spectrum(data).to(self.device)
        metrics = torch.zeros(Ns, Nm).to(self.device)
        for jm, m in enumerate(mlist):
            if m == 'Tpdf':
                metrics[:,jm] = self.calculate_Tges(data)[np.random.randint(0,data.numel(),Ns)]
            if m[:2] == 'ps':
                i = int(m[4:-1])
                metrics[:,jm] = power_spectrum[:,i]
            if m[:2] == 'Mk':
                i = int(m[3])
                tbuf = m[6:-1]
                thr = float(tbuf)
                
                metrics[:,jm] = get_Minkowski(data, i, thr)

        
        return metrics
    
    def get_optimizers(self, gnet, dnet):
        gopt = torch.optim.Adam(gnet.parameters(), lr=self.lr, betas = (0., 0.9))
        dopt = torch.optim.Adam(dnet.parameters(), lr=self.lr, betas = (0., 0.9))
        return gopt, dopt
    
    def get_GAN_nets(self):
        hidden_dim = 64 #?
        gnet = Generator_64(self.latent_dim, hidden_dim).to(self.device)
        dnet = Discriminator_64(64).to(self.device)
        gnet.latent_dim = self.latent_dim
        return gnet, dnet
    
    def calculate_Tges(self, data):
        Tges = data.reshape(-1)
        return Tges
    
    def set_power_spectrum_pars(self):
        npix = 64
        #adapted from https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
        self.kbins = torch.arange(0.5, npix//2+1, 1.)
        kvals = 0.5*(self.kbins[1:] + self.kbins[:-1])
        kfreq = torch.fft.fftfreq(npix)*npix
        kfreq2D = torch.meshgrid(kfreq, kfreq)
        knrm = torch.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        knrm = knrm.flatten()
        #a0, b0 = torch.histogram(knrm, bins=kbins) #get bin counts
        inds, self.arrs, self.Nind = get_knrm_indices(knrm, self.kbins)
        
        
    def calculate_power_spectrum(self, data):
        data = data.cpu() #does not work with cuda?
        Ns = data.shape[0]
        npix = data.shape[-1]

        fft_ges = torch.fft.fft2(data)
        fft_amplitudes_ges = (torch.abs(fft_ges)**2).flatten(start_dim=1)
        ps0_ges = torch.sparse.mm(self.arrs.float(),fft_amplitudes_ges.T).T/self.Nind
        ps_ges = ps0_ges[:,1:-1]* torch.pi * (self.kbins[1:]**2 - self.kbins[:-1]**2)
        
        return ps_ges
        
    #augment data by randomly rotating or flipping Tmaps
    def augment_batch(self, batch, dim_offset = 0, opt = -1):
        if opt == -1:
            opt = np.random.randint(8)
            
        dims = (1+dim_offset,2+dim_offset)
        dims_flip = (0+dim_offset,1+dim_offset)
        
        if opt == 0:
            batch = batch
        elif opt == 1:
            batch = torch.rot90(batch,1,dims=dims)
        elif opt == 2:
            batch = torch.rot90(batch,2,dims=dims)
        elif opt == 3:
            batch = torch.rot90(batch,3,dims=dims)
        elif opt == 4:
            batch = torch.flip(batch,dims_flip)
        elif opt == 5:
            batch = torch.rot90(torch.flip(batch,dims_flip),dims=dims)
        elif opt == 6:
            batch = torch.rot90(torch.flip(batch,dims_flip),2,dims=dims)
        elif opt == 7:
            batch = torch.rot90(torch.flip(batch,dims_flip),3,dims=dims)
    
        return batch

#calculate Minkowski metric imk with threshold thr on batch
def get_Minkowski(batch, imk, thr):
    Ns = batch.shape[0]
    bbuf = batch.cpu().numpy()
    lbuf = bbuf >= thr
    
    res = torch.zeros(Ns)
    for j in range(Ns):
        res[j] = mk.functionals(lbuf[j])[imk]
        
    return res

#calculate bin indices for given component of 2D power spectrum
#when obtaining the 1D power spectrum from the 2D power spectrum
def get_knrm_indices(knrm, kbins):
    knrm_inds = []
    knrm_Nind = []
    knrm_arrs = []
    for j in range(kbins.shape[0]+1):
        if j == 0:
            inds = np.where(knrm < kbins[j])[0]
            arr = (knrm < kbins[j]).int()
        elif j == kbins.shape[0]:
            inds = np.where(knrm > kbins[j-1])[0]
            arr = (knrm > kbins[j-1]).int()
        else:
            buf1 = np.where(knrm > kbins[j-1])[0]
            buf2 = np.where(knrm < kbins[j])[0]
            inds = np.intersect1d(buf1, buf2)
            arr = torch.div((knrm > kbins[j-1]).int() + (knrm < kbins[j]).int(),2, rounding_mode='floor')

        if j == 0:
            knrm_arrs = arr.unsqueeze(0)
        else:
            knrm_arrs = torch.cat((knrm_arrs, arr.unsqueeze(0)), dim=0)
        knrm_inds.append(inds)
        knrm_Nind.append(len(inds))
        #knrm_arrs.append(arr)
        
    knrm_Nind = torch.tensor(knrm_Nind)
    return knrm_inds, knrm_arrs, knrm_Nind   
        
