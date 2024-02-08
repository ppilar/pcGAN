# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import torch
import pickle
#from quantimpy import minkowski as mk

#from .plots import plot_ebm_ax
from ..utils import init_par
from ..Nets import *

import matplotlib.pyplot as plt

class dataset():
    def __init__(self, pars, device):
        self.device = device
        self.init_cp_names()
        self.data = self.generate_data(self.Ns)
        self.constraints = self.calculate_constraints(self.data)
        self.Nc = len(self.constraint_names)
        self.Nebm = self.Nc
        self.cmeans, self.cstds = self.calculate_constraint_stats(self.constraints)
        self.dcov = self.calculate_covariance(self.data)
        self.pmetrics = self.calculate_performance_metrics(self.data)
        self.Np = self.pmetrics.shape[1]
        
        
        self.init_standard_par_values(pars)
        self.latent_dim = pars['ds_latent_dim']
        self.use_cvalue_input = pars['ds_use_cvalue_input']
                
        self.s6_figsize = (20,3)
        self.s16_figsize = (12,12)
        self.bar_width = 0.1
        
        
        self.cinds_selection=-1        
        self.cplot15_bounds = False #use specific bounds for plot
        
    def init_standard_par_values(self, pars):
        init_par(pars, 'lr', 2e-4)
        init_par(pars, 'fsched', 0.2)
        init_par(pars, 'fcov', 1)
        init_par(pars, 'ds_latent_dim', 5)
        init_par(pars, 'ds_use_cvalue_input', False)
        init_par(pars, 'ds_Nc', self.Nc)
        init_par(pars, 'D_batches_together', True)        
        
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
    def initialize_networks(self, pars):
        gnet, dnet = self.get_GAN_nets(pars)
        gopt, dopt = self.get_optimizers(gnet, dnet, pars)
        gsched, dsched = self.get_schedulers(gnet, dnet, gopt, dopt, pars)
        return gnet, dnet, gopt, dopt, gsched, dsched
        
    #initialize generator and discriminator
    def get_GAN_nets(self, pars):
        raise NotImplementedError("Function not implemented!")
        
    #initialize optimizers
    def get_optimizers(self, gnet, dnet, pars):
        gopt = torch.optim.Adam(gnet.parameters(), lr=pars['lr'], betas = (0, 0.9))
        dopt = torch.optim.Adam(dnet.parameters(), lr=pars['lr'], betas = (0, 0.9))
        return gopt, dopt
    
    #initialize learning rate schedulers
    def get_schedulers(self, gnet, dnet, gopt, dopt, pars):
        gsched = torch.optim.lr_scheduler.ExponentialLR(gopt, pars['fsched'])
        dsched = torch.optim.lr_scheduler.ExponentialLR(dopt, pars['fsched'])
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
    
    
    
        