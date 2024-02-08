# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

from .representations import Representation
from ..utils import log_prob_gaussian

class rKDE(Representation):
    def __init__(self, ptrue_rep, fsig=False, include_history=False, fforget=0.9):
        super().__init__(include_history, fforget=fforget)
        
        if type(ptrue_rep) == torch.Tensor:
            self.extract_parameters(ptrue_rep)
            self.initialize_xvecs()
            
            sig = (self.cmaxs - self.cmins)/200
            self.xnet = ptrue_rep.unsqueeze(2)
            self.xsigs = sig.unsqueeze(0).unsqueeze(2)            
            self.pvec0 = self.initialize_pvec()
            
        else:
            if type(fsig) == bool:
                sigs = (ptrue_rep.cstds/ptrue_rep.fsig_best)
            else:
                sigs = ptrue_rep.cstds/fsig
            self.Nc = ptrue_rep.Nc
            self.device = ptrue_rep.device
            self.xsigs = sigs.unsqueeze(0).unsqueeze(2)
            self.xvecs = ptrue_rep.xvecs
        
        
    def update(self, xnet, sigs=False):
        if type(sigs) != bool:
            self.xsigs = sigs.unsqueeze(0).unsqueeze(2)[:,:self.Nc,:]
        self.xnet = xnet.unsqueeze(2)[:,:self.Nc,:]
        self.pvec0 = self.initialize_pvec(full_ds=False)
        
    def initialize_pvec(self, full_ds = True):
        if full_ds == True: #has to be done in loop due to GPU constraints
            pvec = torch.zeros(self.Nc, self.xvecs.shape[1], device=self.device)
            for j in range(self.Nc):
                pvec[j] = self.get_pvec(self.xvecs[j,:], j)
        else:
            pvec = self.get_pvec(self.xvecs.unsqueeze(0))
        return pvec
        
    def get_pvec(self, xvecs='xtrue', jcs = False, weights = False):
        #precalulated pvec at standard x-values
        if type(xvecs) == str:
            if xvecs == 'xtrue':
                pvec = self.pvec0 if jcs == False else self.pvec0[jcs]
                return pvec
            
        #calculate pvec for new x-values
        device0 = xvecs.device
        if type(jcs) == int:
            xvecs = xvecs.unsqueeze(0)
            log_pvec = log_prob_gaussian(xvecs.to(self.xnet.device), self.xnet[:,jcs,:], self.xsigs[:,jcs,:]) #xnet defines means of gaussians
        else:        
            log_pvec = log_prob_gaussian(xvecs.to(self.xnet.device), self.xnet, self.xsigs) #xnet defines means of gaussians
        pvec = torch.exp(log_pvec)
        
        #allow to weight different points individually (e.g. for exponential decay of history)
        if type(weights) == bool:
            pvec = pvec.mean(0)
        else:
            pvec = (pvec*weights.unsqueeze(1).unsqueeze(1)).sum(0)/weights.sum()
            
        return pvec.to(device0)