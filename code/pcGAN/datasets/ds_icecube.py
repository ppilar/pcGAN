# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt


#from .datasets import dataset
from .ds_waveforms import wave_forms
from ..utils import init_par, get_ds_name
from ..Nets import *


#IceCube-Gen2: radio-detector signals
class IceCube_wave_forms(wave_forms):
    def __init__(self, device='cpu', pars=()):
        self.Ns = 50000
        self.init_par_values(pars)
        super().__init__(pars, device)        
        self.ds_name = get_ds_name(pars['ds_opt'])#'IceCube'
        self.cinds_selection = [0,1]#[0,3,5,8,30,50,70]

        
        
    
    #parameters are initialized to the stated values unless they have previously been set to other values
    def init_par_values(self, pars):
        init_par(pars, 'ds_clamp', 0.1)
        init_par(pars, 'lr', 5e-4)
        init_par(pars, 'ds_fKL', 0.1)
        
        
        init_par(pars, 'fsched', 0.5)
        init_par(pars, 'itsched', 40000)
        
        
    
    #choose which constraints to enforce and which high-level features to consider as performance metric
    def init_cp_names(self):
        self.constraint_names = ['max','min']
        self.pmetric_names = ['mean(abs)','max-min','E','N_zc', 'N_max']
    
        
    def generate_data(self, Ns):
        dpath = '../data/'
        fname = 'IceCube.npy'
        
        dbuf = np.load(dpath+fname)
        dbuf = self.remove_invalid_data(dbuf)
        
        dbuf = dbuf/(np.abs(dbuf).max(1))[np.newaxis].T
        ibuf = np.random.choice(np.arange(dbuf.shape[0]), self.Ns)
        Cdata = torch.tensor(dbuf[ibuf], device = self.device).float()
        self.xvec = torch.linspace(0,200,200)
        self.Ns = Cdata.shape[0]
        return Cdata
    
    def get_GAN_nets(self, pars, nopt='big'):
        if pars['GAN_opt'] == 3: 
            dnet = Discriminator_IceCube_v0_SN().to(self.device)
        else:
            dnet = Discriminator_IceCube_v0().to(self.device)            
        gnet = Generator_IceCube_v0(latent_dim = self.latent_dim).to(self.device)
        gnet.xvec = self.xvec
        return gnet, dnet
    
    #remove all-zero signals
    def remove_invalid_data(self, x_trains):
        amplitudes = np.max(np.abs(x_trains),1)
        ibuf = np.where(amplitudes == 0)[0]
        x_trains = np.delete(x_trains, ibuf, 0)
        return x_trains



