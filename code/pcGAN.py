# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from pcGAN.results import Results
from pcGAN.cebm import cEBM
from pcGAN.utils import *
from pcGAN.utils_train import *
from pcGAN.plots import *
from pcGAN.datasets import *


#%% initialize paths, dataset, EBM
if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=0) #set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #select device
if not 'input_path' in locals(): input_path = set_input_path('../results/', 'test') #choose standard folder as input folder, if not otherwise specified
exec(open(input_path+'input.py').read()) #run input file
ds = initialize_ds(ds_opt, device) #initialize dataset
res = Results(ds_opt, fsched = ds.fsched, eval_it = 500, plot_it = 10000, Neval=20000) #initialize res for storing results
rpath, ppath0 = create_dirs(input_path, ds.ds_name) #create folders to store results
cebm, ds = initialize_cebm(ds, ds_opt, bs, res.Neval, ppath0, rpath, load_ds_cebm, device) #initialize conditional EBM
with open(rpath + 'ds'+str(ds_opt)+'_results.pk', 'wb') as f: #save ds + cebm
    pickle.dump([ds, cebm, res],f)
    


#%%
itsched = 1500000
#print parameters
freadme = rpath+'/../readme.txt'
with open(freadme, 'w') as f:
    f.write('GAN_opt:'+str(GAN_opt)+'\n')
    f.write('load_ds_cebm:'+str(load_ds_cebm)+'\n')  
    
    f.write(''+'\n')
    f.write('Nd:'+str(Nd)+'\n') 
    f.write('bs:'+str(bs)+'\n') 
    f.write('Njkl:'+str(Njkl)+'\n') 
    f.write('lr:'+str(ds.lr) + '\n')
    f.write('fKL:'+str(ds.fKL)+'\n')
    f.write('clip:'+str(ds.clamp)+'\n')
    
    f.write('fsched:'+str(ds.fsched)+'\n')
    



#%%

#initialize model-independent paramters
ieval = 0
cval_batches_ges = -1
dcval = -1
gcval = -1
trainloader = DataLoader(ds.data, batch_size=bs*Nd, shuffle=True, num_workers=0)
mstr0 =  'ds' + str(ds_opt) + 'G' + str(GAN_opt) + 'Nd' + str(Nd) + 'bs' + str(bs)


for jm in mode_vec:
    
    #initialize parameters
    itges = 0
    ppath = ppath0 + 'jm' + str(jm)
    mstr = mstr0 + 'm'+str(jm)+'_'    
    res.losses[jm] = [[] for j in range(4)]
    dloss0, dloss, dloss_GP, gloss, loss_kl = torch.tensor([0,0,0,0,0])    
    gsin = get_randn(16, device, ds.latent_dim)
    gnet, dnet, gopt, dopt, gsched, dsched = ds.initialize_models()
    akls = np.ones(ds.Nc)
    awkl = update_wvec(akls)
    
        
    #%% training loop
    gnet.train()
    dnet.train()
    
    for epoch in range(99999):
        for i, batch_ges in enumerate(trainloader):
            ti = time.time()
            
            #####
            ##### update discriminator
            
            # standard discriminator loss
            if jm != 2:
                for jD in range(Nd):
                    #determine batch size
                    bsD = min((jD+1)*bs, batch_ges.shape[0]) - jD*bs
                    if bsD <= 0:
                        break
                    
                    #select batch and generate fake batch
                    batch = batch_ges[jD*bs:min((jD+1)*bs,bs*Nd)]
                    batch = ds.augment_batch(batch)
                    
                    gin = get_randn(bsD, device, ds.latent_dim)
                    gbatch = gnet(gin)
                    
                    
                    #calculate loss and update D - should be possible to concatenate both?
                    dopt.zero_grad()
                    
                    batches_ges = torch.cat((batch, gbatch),0)
                    labels_ges = torch.cat((torch.ones(bsD, device=device), torch.zeros(bsD, device=device)))
                    if ds_opt == 2:
                        dbuf1 = dnet((batch, dcval))
                        dbuf2 = dnet((gbatch, gcval))
                        dbuf = torch.cat((dbuf1, dbuf2),0)
                    else:
                        dbuf = dnet((batches_ges, cval_batches_ges))
                    dloss_true = dbuf[:bsD]
                    dloss_gen = dbuf[bsD:2*bsD]                    
                    
                    dloss0 = torch.mean(dloss_gen - dloss_true)
                    
                    if GAN_opt == 2: #WGAN-GP
                        lGP = 10
                        
                        eps = torch.rand(bsD).unsqueeze(1).to(device).unsqueeze(2)
                        xhat = eps*batch + (1-eps)*gbatch.detach().squeeze()                        
                        xhat.requires_grad = True
                        Dxhat = dnet((xhat, dcval))
                        
                        Dxhat_prime = torch.autograd.grad(torch.sum(Dxhat), xhat, create_graph=True, retain_graph=True)[0]
                        dloss_GP = lGP*torch.mean((torch.norm(Dxhat_prime,dim=1) - torch.tensor(1.))**2)
                        dloss = dloss0 + dloss_GP
                    elif GAN_opt == 0: #GAN                   
                        dloss0 = nn.BCELoss()(torch.sigmoid(dbuf), labels_ges)
                        dloss = dloss0
                    else: #WGAN
                        dloss = dloss0
                    
                    dloss.backward()
                    dopt.step()
                    
                        
                    if GAN_opt == 1: #clamp discriminator weights for WGAN                   
                        with torch.no_grad():
                            for param in dnet.parameters():
                                param.clamp_(-ds.clamp, ds.clamp)
                                
                    
            
            #####
            ##### update generator
            
            # standard generator loss
            gin = get_randn(bs, device, ds.latent_dim)
            gbatch = gnet(gin)
                
            gopt.zero_grad()
            
            gbuf = dnet((gbatch, gcval))
            if GAN_opt == 0:
                labels_gen = torch.ones(bs, device=device)
                gloss0 = nn.BCELoss()(torch.sigmoid(gbuf), labels_gen)
            else:
                gloss0 = -torch.mean(gbuf) if jm != 2 else torch.tensor(0.)
            
            
            #add probabilistic constraint to generator loss
            if jm in [1,2] and itges >= 0:
                cval_batch = ds.calculate_constraints(gbatch)
                cmeans, cstds = ds.calculate_constraint_stats(cval_batch)
                
                #calculate KL for all constraints
                tKL = time.time()
                loss_kls = calculate_multiple_KL(cebm, cval_batch[:,:ds.Nebm], fsig=cebm.fsig_best, cut_tails = ds.cut_tails)
                
                #add KL for randomly selected constraints to loss
                #Njkl = 3
                loss_kl = 0
                for k in range(Njkl):
                    jkl = np.random.choice(range(ds.Nc), 1, p=awkl).item()
                    akls[jkl] = (akls[jkl] + loss_kls[jkl].item())/2
                    awkl = update_wvec(akls)
                    loss_kl = loss_kl + loss_kls[jkl]*ds.fKL/Njkl
                    
                gloss = gloss0 + loss_kl
                
                
                #loss_kl = ds.fKL*loss_kls.mean()
                #gloss = gloss0 + loss_kl#loss_kls.mean()
            elif jm == 3:
                mbcov = ds.calculate_covariance(gbatch)
                loss_kl = ds.fcov*torch.linalg.matrix_norm(ds.dcov-mbcov)
                gloss = gloss0 + loss_kl
            else:
                gloss = gloss0
            
            gloss.backward()
            gopt.step()
            
            #####
            #####
            # display and gather results
            
            print_and_update(res, jm, epoch, itges, ti, dloss0, dloss_GP, gloss0, loss_kl)
            if itges >= itmax:
                break            
            
            itges += 1        
            if itges%res.eval_it == 0 or itges == 1:
                res, ieval = plot_and_evaluate(ds, res, cebm, jm, itges, dnet, gnet, gsin, gcval, ppath, mstr, device, ieval, res.Neval)

            if (itges==itsched):#itges%itsched == 0): #waveform
                dsched.step()
                gsched.step()
                
                with open(freadme, 'a') as f:
                    f.write('lr changed at it ='+str(itges)+'\n')
            # if (itges%100000 == 0 or itges%150000 == 0):
            #     dsched.step()
            #     gsched.step()
        
        if itges >= itmax:
            break
    #%%
    plot_losses(res.losses[jm], ppath+mstr+'losses.pdf')
    res.gnet_ges[jm].append(gnet)
    res.dnet_ges[jm].append(dnet)
    
    
#%%    
with open(rpath + 'ds'+str(ds_opt)+'_results.pk', 'wb') as f:
    pickle.dump([ds, cebm, res],f)
  