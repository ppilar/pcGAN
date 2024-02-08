# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
from .utils import get_randn, update_wvec, check_dirs, get_ds_name


#class to store results
class Results():
    def __init__(self, pars, input_path, eval_it = 500, plot_it = 2500, schedule_it = 100000, Neval = 20000, device='cpu'):
        self.pars = pars
        self.rpath, self.ppath0 = self.create_dirs(input_path, get_ds_name(pars['ds_opt']))
        self.device = device
        
        self.initial_pars = dict()
        self.Nmodels = max(7, max(pars['model_vec'])) + 1 
        self.Neval = Neval
        self.mname_vec = [[] for j in range(self.Nmodels)]
        self.mname_vec[0] = 'WGAN'
        self.mname_vec[1] = 'pcGAN'
        self.mname_vec[2] = 'KL'
        self.mname_vec[3] = 'Wu et al.'
        self.mname_vec[4] = 'WGAN-GP'
        self.mname_vec[5] = 'SNGAN'
        
        Nrun = pars['Nrun']
        self.losses_ges = [[] for j in range(Nrun)]
        self.constraints_hist_abs_ges_ges = [[] for j in range(Nrun)]
        self.constraints_hist_KL_ges_ges = [[] for j in range(Nrun)]
        self.constraints_ges_ges = [[] for j in range(Nrun)]
        self.constraints_delta_ges_ges = [[] for j in range(Nrun)]
        self.pmetrics_hist_abs_ges_ges = [[] for j in range(Nrun)]
        self.pmetrics_hist_KL_ges_ges = [[] for j in range(Nrun)]
        self.gnet_ges_ges =  [[] for j in range(Nrun)]
        self.dnet_ges_ges =  [[] for j in range(Nrun)]
        self.ccount_ges_ges = [[] for j in range(Nrun)]
        
        self.eval_it = eval_it
        self.plot_it = plot_it
        self.schedule_it = schedule_it
        self.fsched = pars['fsched']
    
    def create_dirs(self, input_path, ds_name):
        dpath = input_path + ds_name + '/'
        rpath = dpath + 'files/'
        ppath0 = dpath + 'plots/'
        
        
        check_dirs(input_path, dpath)    
        check_dirs(input_path, ppath0)
        check_dirs(input_path, rpath)

        return rpath, ppath0
        
        
    def init_run_results(self):
        self.losses = [[] for j in range(self.Nmodels)]
        self.constraints_hist_abs_ges = [[] for j in range(self.Nmodels)]
        self.constraints_hist_KL_ges = [[] for j in range(self.Nmodels)]
        self.constraints_ges = [[] for j in range(self.Nmodels)]
        self.constraints_delta_ges = [[] for j in range(self.Nmodels)]  
        self.pmetrics_hist_abs_ges = [[] for j in range(self.Nmodels)]      
        self.pmetrics_hist_KL_ges = [[] for j in range(self.Nmodels)]
        self.gnet_ges =  [[] for j in range(self.Nmodels)]
        self.dnet_ges =  [[] for j in range(self.Nmodels)]
        self.ccount_ges = [[] for j in range(self.Nmodels)]
    
    def store_run_results(self, jN):
        self.losses_ges[jN] = self.losses
        self.constraints_hist_abs_ges_ges[jN] = self.constraints_hist_abs_ges
        self.constraints_hist_KL_ges_ges[jN] = self.constraints_hist_KL_ges
        self.constraints_ges_ges[jN] = self.constraints_ges
        self.constraints_delta_ges_ges[jN] = self.constraints_delta_ges
        self.pmetrics_hist_abs_ges_ges[jN] = self.pmetrics_hist_abs_ges
        self.pmetrics_hist_KL_ges_ges[jN] = self.pmetrics_hist_KL_ges
        self.gnet_ges_ges[jN] =  self.gnet_ges
        self.dnet_ges_ges[jN] =  self.dnet_ges
        self.ccount_ges_ges[jN] = self.ccount_ges
        
    def init_model_independent(self):
        self.use_fEBM = False
        self.ieval = 0
        self.cval_batches_ges = -1
        self.dcval = -1
        self.gcval = -1
        self.mstr0 =  'ds' + str(self.pars['ds_opt']) + 'G' + str(self.pars['GAN_opt']) + 'Nd' + str(self.pars['Nd']) + 'bs' + str(self.pars['bs'])

        
    def init_model_results(self, jN, jm, Nc, latent_dim):
        self.itges = 0
        
        self.ppathr = self.ppath0 + 'Run' + str(jN) + '/'
        self.ppathr_it = self.ppathr + 'training/'        
        check_dirs(self.ppath0, self.ppathr, copy_input = False)
        check_dirs(self.ppathr, self.ppathr_it, copy_input = False)
        self.mstr = 'm'+str(jm) + '_'
        
        self.ppathr = self.ppathr + self.mstr
        self.ppathr_it = self.ppathr_it + self.mstr
        
        self.losses[jm] = [[] for j in range(4)]
        self.dloss0, self.dloss, self.dloss_GP, self.gloss, self.loss_kl = torch.tensor([0,0,0,0,0])    
        self.gsin = get_randn(16, self.device, latent_dim)
        self.akls = torch.ones(Nc)
        self.awkl = update_wvec(self.akls)
        self.ccount_ges[jm] = np.zeros(Nc)