# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import scipy as sp
import sklearn as skl
import matplotlib.pyplot as plt
import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from pcGAN.results import Results
from pcGAN.utils import *
from pcGAN.plots import *
from pcGAN.datasets import *
from pcGAN.representations import *
from pcGAN.utils_plot_results import *
from pcGAN.utils_eval_results import *



#%%
def normalize_metrics(metric):
    return (metric - metric.mean(0))/metric.std(0)

   
#%%
rand_init, s = init_random_seeds(s=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ds_opt = 1
if ds_opt == 1:
    #opt_list =  ['model_comp', 'bs_comp', 'match_opt_comp', 'omega_comp']
    opt_list = ['model_comp']
if ds_opt == 3:
    opt_list = ['model_comp']
    
    
for opt in opt_list:
    
    ds, ptrue_rep, res = load_run_data(ds_opt, opt = opt)
    if ds_opt == 3:
        ds.pmetric_names = ['mean(abs)','max-min','E','N_zc', 'N_max']
        ds.pmetrics = ds.calculate_metrics(ds.data, ds.pmetric_names)
    ppath = '../plots/ds'+str(ds_opt)+'_'+opt+'_'
    Nrun = res.Nrun
    #%%
    
    k = 10
    dsamples_pm = normalize_metrics(np.array(ds.calculate_metrics(ds.data, ds.pmetric_names).cpu()))
    kdist_pm = get_kdist(dsamples_pm[:20000], k=k)
    
    #%%
    Npmetrics = len(ds.pmetric_names)
    Nmodels = len(res.model_vec)
    mnames = []
    
    #initialize arrays
    FID_ges, precision_ges, recall_ges, F1_ges = np.zeros((4, Nrun, Nmodels))
    pmetrics_ges = np.zeros((Nrun, Nmodels, Npmetrics))
    loss_KL_ges = np.zeros((Nrun, Nmodels, ds.Nc))
    
    #%%    
    jm_order = range(len(res.model_vec))
    for jN in range(Nrun):
        for j, jm in enumerate(jm_order):
        #for j, jm in enumerate(res.model_vec):
            write_and_flush('jN='+str(jN)+', jm='+str(jm))
            
            #generate data
            gnet = res.Gges[jN][jm][0]
            gges = generate_Ns(gnet, 20000, device, gnet.latent_dim).squeeze()
            
            #various FID scores
            FID_ges[jN, j] = get_FID(ds.data, gges)      
            
            #precision and recall for performance metrics and constraints
            gsamples_pm = normalize_metrics(np.array(ds.calculate_metrics(gges, ds.pmetric_names).cpu()))
            precision_ges[jN, j], _ = get_precision(gsamples_pm, dsamples_pm[:20000], kdist_pm[:20000])
            recall_ges[jN, j], _ = get_recall(gsamples_pm, dsamples_pm[:20000], k=k)
            F1_ges[jN, j] = 2*precision_ges[jN, j]*recall_ges[jN, j]/(precision_ges[jN, j] + recall_ges[jN, j])
            pmetrics_ges[jN, j, :] = plot_pmetrics(ds, gges, '', 'abs', plot=False)[1]
            loss_KL_ges[jN, j, :] = plot_constraints(ds, gges, ptrue_rep, '', 'abs', plot=False, all_constraints=True)[1]
            
            
            if jN == 0:
                mnames.append(res.mname_vec2[jm])            
    
    #%% boxplot of metrics
    metric_list =[FID_ges, F1_ges, loss_KL_ges[:,:].mean(-1), pmetrics_ges.mean(-1)]
    name_list = [r'$d_F^2$', r'$F_1$', r'$\bar V_c$', r'$\bar V_f$']
    plot_box_plots(metric_list, name_list, mnames, pname=ppath + 'eval_selection')
    
    #%% boxplot of small selection
    metric_list =[FID_ges, loss_KL_ges[:,:].mean(-1)]
    name_list = [r'$d_F^2$', r'$\bar V_c$']
    plot_box_plots(metric_list, name_list, mnames, pname=ppath + 'eval_selection_small')

