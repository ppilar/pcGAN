# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from torch.utils.data import DataLoader


from pcGAN.results import Results
from pcGAN.utils import *
from pcGAN.utils_train import *
from pcGAN.plots import plot_losses
from pcGAN.representations.rKDE import rKDE





#%% initialize paths, dataset, EBM
if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=0) #set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #select device
#device='cpu'
if not 'input_path' in locals(): input_path = set_input_path('../results/', 'test') #choose standard folder as input folder, if not otherwise specified
exec(open(input_path+'input.py').read()) #run input file
initialize_standard_pars(pars)

ppath0 = '../plots/'
fpath0 =  '../results/datasets/' #to save ds and ptrue
ds = initialize_ds(pars, device, fpath0) #initialize dataset
res = Results(pars, input_path, eval_it = 500, plot_it = 3000, Neval = pars['Neval'], device=device) #initialize object to store results
ptrue_rep = initialize_ptrue_rep(ds, pars, ppath0, fpath0, plot=False) #initialize representation of true data distributions
pfake_rep = rKDE(ptrue_rep, include_history = pars['include_history'], fforget = pars['fforget']) #initialize representatino of generated data distributions

print_parameters(pars, res.rpath + '/../standard_pars.txt')
#%%
res.init_model_independent()
trainloader = DataLoader(ds.data, batch_size=pars['bs']*pars['Nd'], shuffle=True, num_workers=0)


for jN in range(pars['Nrun']):
    print('Run ' + str(jN))
    res.init_run_results()
    
    for jm in pars['model_vec']:        
        #initialize tensorboard writer, parameters and nets
        writer, pbuf, fpath = initialize_model_parameters(res, pars, comment, jN, jm)
        gnet, dnet, gopt, dopt, gsched, dsched = ds.initialize_networks(pars)
        
            
        #%% training loop
        gnet.train()
        dnet.train()
        
        for epoch in range(99999):
            for i, batch_ges in enumerate(trainloader):
                ti = time.time()
                              
                ### update discriminator
                #if jm != 2:
                if pars['use_dloss'] == True:
                    for jD in range(pars['Nd']):
                        dnet, dpot, flag = discriminator_step(ds, pars, res, dnet, gnet, dopt, batch_ges, jD)
                        if flag: break                                
                        
                
                ### update generator
                    
                #get generator loss
                res.gloss0, gbatch = get_gloss0(dnet, gnet, pars, res.gcval, device)
                res.loss_kl = get_gloss_constraint(res, pars, ptrue_rep, pfake_rep, jm, ds, gbatch)
                res.gloss = res.gloss0 + pars['omega_c']*res.loss_kl                
                               
                gopt.zero_grad()
                res.gloss.backward()
                gopt.step()
                 
                ### display and gather results                
                print_and_update(writer, res, pars, ptrue_rep, jm, epoch, res.itges, ti)    
                if res.itges%res.eval_it == 0:# or res.itges == 1:
                    plot_and_evaluate(ds, res, writer, ptrue_rep, pfake_rep, pars, jm, dnet, gnet, device)
                    
                
                ### increase iteration
                res.itges += 1
                if res.itges > pars['itmax']: break
                if (res.itges%pars['itsched'] == 0 and res.itges > 1): #TODO: fix pcGAN scheduler
                    scheduler_step(dsched, gsched, fpath, res.itges)
                
            if res.itges >= pars['itmax']:
                break
        #add evaluation metrics to writer
        writer.add_hparams(
            pbuf,
            {'constraints: values': np.array(res.constraints_ges[jm][-1]).mean(),
             'constraints: deltas': np.array(res.constraints_delta_ges[jm][-1]).mean(),
             'FID: samples': res.FID,
             'FID: constraints': res.cFID,
             'FID: pmetrics': res.pFID
             })
        #%%
        #plot losses
        plot_losses(res.losses[jm], res.ppathr+res.mstr+'losses.pdf')
        
        #add nets to results
        res.gnet_ges[jm].append(gnet)
        if jm == 5 or jm == 7:
            res.dnet_ges[jm].append(dnet.state_dict())
        else:
            res.dnet_ges[jm].append(dnet)
        writer.close()
        
    res.store_run_results(jN)
    

    
#%% save results
with open(res.rpath + 'ds'+str(pars['ds_opt'])+'_results.pk', 'wb') as f:
    pickle.dump([res, pars], f)
  

    
    