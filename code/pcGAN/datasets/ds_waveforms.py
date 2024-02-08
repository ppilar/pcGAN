# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt

from .datasets import dataset
from ..utils import init_par, get_ds_name
from ..Nets import *
   
#synthetic example: dataset of waveforms, i.e. superpositions of 2 sine waves
class wave_forms(dataset):
    def __init__(self, pars, device='cpu'):
        self.Ns = 100000
        self.init_par_values(pars)
        super().__init__(pars, device)
        
        
        self.sample_ptype = plt.plot
        self.ds_name = get_ds_name(pars['ds_opt'])
        self.cinds_selection = [0,1,2,5,10,15,30,50,75,100]
        
        self.s16_figsize = (12,13.5)
        self.plot_fft = False
        
    
    #parameters are initialized to the stated values unless they have previously been set to other values
    def init_par_values(self, pars):
        print('3')
        init_par(pars, 'ds_clamp', 5e-3)
        init_par(pars, 'lr', 2e-4)
        
        init_par(pars, 'ds_fKL', 100)
        init_par(pars, 'ds_fKLG', 50/6)
        init_par(pars, 'ds_fm', 1000*3/2)
        
        
        init_par(pars, 'fsched', 0.5)
        init_par(pars, 'itsched', 70000)
        
    #choose which constraints to enforce and which high-level features to consider as performance metric
    def init_cp_names(self):
        Nc = 101
        self.constraint_names = ['ps [' + str(j) + ']' for j in range(Nc)]
        self.pmetric_names = ['mean','mean(abs)','max-min','E','N_zc', 'N_max']
        
    def generate_data(self, Ns):
        Nf = 2 #also change output layer in Net!
        lx=20
        Nx=200        
        nf=0
        
        xvec = np.linspace(0,lx,Nx)
        
        fs_ges = np.abs(np.random.normal(1,1,Nf*Ns)).reshape(Ns,Nf)
        fs_ges = np.expand_dims(fs_ges,2)

        wvec_ges = np.sum(np.sin(np.kron(fs_ges, xvec)), 1)
        if nf > 0:
            nvec_ges = np.random.normal(0,nf, wvec_ges.size).reshape(wvec_ges.shape)
            wvec_ges += nvec_ges
            
        
        self.xvec = torch.tensor(xvec)
        return torch.tensor(wvec_ges/Nf, device=self.device).float()
    
    #how to calculate various metrics
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
            if m == 'N_zc':
                metrics[:,jm] = ((data[:,:-1] * data[:,1:]) < 0).sum(1)
            if m == 'N_max':
                diff = torch.diff(data,1)
                metrics[:,jm] = ((diff[:,:-1] * diff[:,1:]) < 0).sum(1)
                
        
        return metrics
    
    #initialize nets
    def get_GAN_nets(self, pars, nopt='big'):
        if pars['GAN_opt'] == 3: #spectral normalization
            dnet = Discriminator_SN_1D_large(ds=1, nopt = nopt, use_fft_input = pars['ds_use_cvalue_input']).to(self.device)            
        else:
            dnet = Discriminator_1D_large(ds=1, nopt = nopt, use_fft_input = pars['ds_use_cvalue_input']).to(self.device)        
        gnet = Generator_1D_large(ds=1, nopt = nopt, latent_dim = pars['ds_latent_dim'], device=self.device).to(self.device)
        gnet.xvec = self.xvec        
        
        return gnet, dnet
        
    
#calculate real discrete Fourier transform
def get_fft(wf, backend='torch'):
    if backend == 'torch':
        fftges = torch.fft.rfft(wf, dim=1)
        #fftabs = get_abs(fftges)
        fftabs = torch.abs(fftges)
        fftmean = torch.mean(fftabs,0)
        fftstd = torch.std(fftabs,0)
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