# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import scipy as sp
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
from pcGAN.utils_plot_results import *
from pcGAN.utils_eval_results import *

#%%
rand_init, s = init_random_seeds(s=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = 'model_comp'
ds_opt = 1
ds, ptrue_rep, res = load_run_data(ds_opt, opt = opt)
Nrun = res.Nrun
ppath = '../plots/ds'+str(ds_opt)+'_'+opt+'_'


#%% generate data for the different GANs
jN = 0
gges = []
mnames = []

    
jm_order = range(len(res.model_vec))
for jm in jm_order:
    gnet = res.Gges[jN][jm][0]
    gges.append(generate_Ns(gnet, 20000, device, gnet.latent_dim).squeeze())
    mnames.append(res.mname_vec2[jm])
    


#%% plot comparison of constraint fulfillment
atitle = ['real'] + mnames
ds.cinds_selection = res.cinds_selection
plot_constraints(ds, gges, ptrue_rep, ppath, 'abs', plot=True, atitle=atitle, all_constraints = False, crange_opt = 'plot')

#%% plot real and generated samples
plot_sample_comparison(ds, gges, ds_opt, atitle, ppath)














