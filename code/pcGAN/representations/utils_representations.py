# -*- coding: utf-8 -*-

import torch
import numpy as np
from ..utils import write_and_flush, calculate_dist_loss



#sample minibatch
def sample_mb(ds, bs=64):
    perm = torch.randperm(ds.data.size(0))
    idx = perm[:bs]
    return ds.data[idx], ds.constraints[idx]

def get_p_representations(ptrue_rep, pfake_rep, xvecs='xtrue'):
    pvec_true = ptrue_rep.pvec(xvecs)
    pvec_fake = pfake_rep.pvec(xvecs)
    if xvecs == 'xtrue':
        xvecs = ptrue_rep.xvecs
    
    return pvec_true, pvec_fake, xvecs


def calculate_KLs_for_fsig(ds, ptrue_rep, match_opt, bs, Nsig = 200, Navg = 50):
    from .rKDE import rKDE

    fsig_vec = np.logspace(-1,2,Nsig)
    KLs = torch.zeros(ptrue_rep.Nc, Nsig, Navg)
    KLs2 = torch.zeros(ptrue_rep.Nc, Nsig, Navg)
    for jN in range(Navg):
        write_and_flush(str(jN))
        
        if bs < 3000:
            use_weights = ptrue_rep.include_history# if bs < 3000 else False 
            if ptrue_rep.include_history == 1:
                Nb = len(ptrue_rep.hfacs_exp) 
            elif ptrue_rep.include_history == 2:
                Nb = len(ptrue_rep.hfacs) 
            else:
                Nb = 1 #TODO: extract Nb from rfake
        else:
            use_weights = False
            Nb = 1
        
        
        mb, cmb = sample_mb(ds, Nb*bs)
        for jsig, fsig in enumerate(fsig_vec):
            rfake = rKDE(ptrue_rep, fsig)
            rfake.update(cmb[:,:])
            if use_weights:
                buf_weights = rfake.hfacs_exp if use_weights == 1 else rfake.hfacs
                weights = torch.tensor(buf_weights).repeat(bs,1).T.reshape(-1).to(ptrue_rep.device)
            else:
                weights = False
            pvec_true = ptrue_rep.get_pvec()
            pvec_fake = rfake.get_pvec(ptrue_rep.xvecs.unsqueeze(0), weights = weights)
            buf = calculate_dist_loss(pvec_true, pvec_fake, ptrue_rep.xvecs, match_opt = match_opt).detach().cpu()
            KLs[:, jsig, jN] = buf
            
    return KLs, fsig_vec


def calculate_KLs_for_real_data(ds, ptrue_rep, match_opt, bs):
    KLs = torch.zeros(Nc, Navg)
    for jN in range(Navg):
        write_and_flush(str(jN))
        
        mb, cmb = sample_mb(ds, Neval)            
        rfake = rKDE(ptrue_rep, fsig_vec)
        rfake.update(cmb[:,:])
        pvec_true = ptrue_rep.get_pvec()
        pvec_fake = rfake.get_pvec(ptrue_rep.xvecs.unsqueeze(0))            
        KLs[:, jN] = calculate_dist_loss(pvec_true, pvec_fake, ptrue_rep.xvecs, match_opt = match_opt)
        
    return KLs