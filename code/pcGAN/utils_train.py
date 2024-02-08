# -*- coding: utf-8 -*-

import os

import random
import shutil
import time
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
from .Nets import *
from .datasets import *
from .utils import *
from .plots import plot_losses, plot_metric_trajectories, plot_summary, plot_rep, plot_pmetrics, plot_metrics, plot_constraints, plot_samples, plot_constraint_fulfillment
from .representations.representations import get_p_representations
from .utils_eval_results import *
from .representations.rKDE import rKDE

from torch.utils.tensorboard import SummaryWriter


#initialize dataset according to ds_opt
def initialize_ds(pars, device, path0):
    if path0 != '':
        check_dirs('', path0, copy_input=False)
    fpath = path0 + 'ds'+str(pars['ds_opt'])+'.pk'
    file_exists = os.path.isfile(fpath)
    
    if pars['load_ds'] == 1 and file_exists:
        with open(fpath, 'rb') as f:
            ds = pickle.load(f)
        ds.init_par_values(pars)
        ds.init_standard_par_values(pars)    
    else:
        if pars['ds_opt'] == 1:
            ds = wave_forms(pars=pars, device=device)
        if pars['ds_opt'] == 2:
            ds = CAMELS(pars=pars, device=device)
        if pars['ds_opt'] == 3:
            ds = IceCube_wave_forms(pars=pars, device=device)
        with open(fpath, 'wb') as f:
            pickle.dump(ds, f)
        
        
    return ds

#initialize representation of the true distribution
def initialize_ptrue_rep(ds, pars, ppath0, fpath0, update_fsig = True, plot=True):
    if fpath0 != '':
        check_dirs('', fpath0, copy_input=False)
    if ppath0 != '':
        check_dirs('', ppath0, copy_input=False)    
    fpath = fpath0 + 'ds'+str(pars['ds_opt'])+'_ptrue_'+pars['ptrue_rep'] +'.pk'
    
    #represent true distribution via KDE
    if pars['ptrue_rep'] == 'KDE':
        file_exists = os.path.isfile(fpath)        
        if pars['load_ptrue_rep'] == 1 and file_exists:
            with open(fpath, 'rb') as f:
                ptrue_rep = pickle.load(f)
        else:
            ptrue_rep = rKDE(ds.constraints)
            ptrue_rep.initialize_real_data_rep(pars, ds.constraints)
    
    #plot representation of the distributions in case that it is newly created
    if pars['load_ptrue_rep'] == 0 and plot:
        plot_rep(ptrue_rep, ds.constraints[:,:ds.Nc], ds.constraint_names, ppath0)
            
    #update current bs and match_opt together with optimal values for fsig
    if update_fsig:
        ptrue_rep.check_match_bs(ds, pars['match_opt'], pars['bs'])
        
    with open(fpath, 'wb') as f:
        pickle.dump(ptrue_rep, f)
        
    return ptrue_rep


#initialize parameters and bookkeeping devices for current run of model
def initialize_model_parameters(res, pars, comment, jN, jm):
    reset_to_initial(pars, res) #reset parameters to model-independent values
    update_model_pars(jm, res, pars) #update model specific parameters
    fpath = res.rpath + '/../m'+str(jm)+'_pars.txt'
    print_parameters(pars, fpath) #print parameters to text file
    res.init_model_results(jN, jm, pars['ds_Nc'], pars['ds_latent_dim']) #initialize arrays in res for storing results
    
    #initialize tensorboard writer
    if pars['par_label'] != 'none':
        comment = pars['match_opt'] + comment
    else:
        comment = pars['match_opt']
    writer = initialize_writer("../runs/runs_pcgan", comment0 = 'ds' + str(pars['ds_opt']) + 'm'+str(jm), comment = comment)
    print_parameters(pars, writer.log_dir + '/pars.txt')
    #initialize tensorboard hparams
    pbuf = pars.copy()
    pbuf['model_vec'] = None
    writer.add_hparams(pbuf,
    {'constraints: values': 1,
      'constraints: deltas': 1,
      'FID: samples': 1,
      'FID: constraints': 1,
      'FID: pmetrics': 1}
    )
    
    return writer, pbuf, fpath

#take one step with discriminator
def discriminator_step(ds, pars, res, dnet, gnet, dopt, batch_ges, jD):
    #determine batch size
    bsD = min((jD+1)*pars['bs'], batch_ges.shape[0]) - jD*pars['bs']
    flag = True if bsD <= 0 else False
    if flag:
        return dnet, dopt, flag
    
    #calculate batches and loss
    batches_ges, labels_ges, cval_batches_ges = get_dstep_batches(ds, gnet, batch_ges, jD, pars['bs'], bsD, pars['Nd'])
    res.dloss, res.dloss0 = get_dloss(dnet, batches_ges, labels_ges, cval_batches_ges, bsD, jD, pars['GAN_opt'], pars['D_batches_together'], pars['sn_dloss_max'])
                            
    #update
    dopt.zero_grad()                    
    res.dloss.backward()
    dopt.step()                        
        
    #clamp discriminator weights for WGAN   
    if pars['GAN_opt'] == 1:                 
        with torch.no_grad():
            for param in dnet.parameters():
                param.clamp_(-pars['ds_clamp'], pars['ds_clamp'])
                
    return dnet, dopt, flag

#batches involved in calculating dloss
def get_dstep_batches(ds, gnet, batch_ges, jD, bs, bsD, Nd):
    batch = batch_ges[jD*bs:min((jD+1)*bs,bs*Nd)]
    batch = ds.augment_batch(batch)
    
    device = batch.device               
    gin = get_randn(bsD, device, ds.latent_dim)
    gbatch = gnet(gin)
    
    batches_ges = torch.cat((batch, gbatch),0)
    labels_ges = torch.cat((torch.ones(bsD, device=device), torch.zeros(bsD, device=device)))
    cval_batches_ges = -1#*torch.ones(batches_ges.shape[0], device=device) #placeholder to allow for conditioning later on
    return batches_ges, labels_ges, cval_batches_ges


#calculate discriminator loss
def get_dloss(dnet, batches_ges, labels_ges, cval_batches_ges, bsD, jD, GAN_opt, D_batches_together, sn_dloss_max):
  
    dloss_true, dloss_gen = get_dloss_true_gen(dnet, batches_ges, cval_batches_ges, bsD, D_batches_together)    
    dloss0 = torch.mean(dloss_gen - dloss_true)
    
    if GAN_opt == 2: #WGAN-GP
        lGP = 10
        #dcval = cval_batches_ges[:bsD]
        dcval = cval_batches_ges
        batch = batches_ges[:bsD,:].detach()
        gbatch = batches_ges[bsD:,:]
        
        eps = torch.rand(bsD).unsqueeze(1).to(batches_ges.device)
        xhat = eps*batch + (1-eps)*gbatch.detach().squeeze()                        
        xhat.requires_grad = True
        Dxhat = dnet((xhat, dcval))
        
        Dxhat_prime = torch.autograd.grad(torch.sum(Dxhat), xhat, create_graph=True, retain_graph=True)[0]
        dloss_GP = lGP*torch.mean((torch.norm(Dxhat_prime,dim=1) - torch.tensor(1.))**2)
        dloss = dloss0 + dloss_GP
    elif GAN_opt == 0: #GAN                   
        dloss0 = nn.BCELoss()(torch.sigmoid(dbuf).squeeze(), labels_ges)
        dloss = dloss0
    elif GAN_opt == 3: #SNGAN
        dloss = dloss0 + torch.max(torch.tensor(0), torch.abs(dloss0) - torch.tensor(sn_dloss_max))**2 #to avoid extremely large values
    else: #WGAN
        dloss = dloss0
        
    return dloss, dloss0

#calculates discriminator prediction; batches can be passed individually or together
def get_dloss_true_gen(dnet, batches_ges, cval_batches_ges, bsD, pass_together = True):
    if pass_together == False:
        batch = batches_ges[:bsD,:]
        gbatch = batches_ges[bsD:,:]
        
        if type(cval_batches_ges) != int:
            dcval = cval_batches_ges[:bsD,:]
            gcval = cval_batches_ges[bsD:,:]
        else:
            dcval, gcval = -1, -1
        
        dbuf1 = dnet((batch, dcval))
        dbuf2 = dnet((gbatch, gcval))
        dbuf = torch.cat((dbuf1, dbuf2),0)
    else:
        dbuf = dnet((batches_ges, cval_batches_ges))
    
    dloss_true = dbuf[:bsD]
    dloss_gen = dbuf[bsD:2*bsD]
    return dloss_true, dloss_gen

#calculate standard generator loss
def get_gloss0(dnet, gnet, pars, gcval, device):
    gin = get_randn(pars['bs'], device, pars['ds_latent_dim'])
    gbatch = gnet(gin)
    
    gbuf = dnet((gbatch, gcval))
    if pars['use_gloss'] == True:
        if pars['GAN_opt'] == 0:#  or GAN_opt == 3:
            labels_gen = torch.ones(pars['bs'], device=device)
            gloss0 = nn.BCELoss()(torch.sigmoid(gbuf).squeeze(), labels_gen)
        else:
            gloss0 = -torch.mean(gbuf) #if jm != 2 else torch.tensor(0.)
    else:
        gloss0 = torch.tensor(0)
        
    return gloss0, gbatch


#calculate constraint terms for generator loss
def get_gloss_constraint(res, pars, ptrue_rep, pfake_rep, jm, ds, gbatch):
    if pars['use_pc_loss'] == True:       
        tKL = time.time()
         
        #calculate difference in the distributions according ot the chosen measure
        cval_gbatch = ds.calculate_constraints(gbatch)
        pfake_rep.update(cval_gbatch[:,:ds.Nc])
        pvec_true, pvec_fake, xvecs = get_p_representations(ptrue_rep, pfake_rep, 'xtrue')
        
        #loss for all constraints and deltas with optimal values as obtained with real data
        loss_kls = calculate_dist_loss(pvec_true, pvec_fake, xvecs[:ds.Nc,:], match_opt = pars['match_opt'])        
        delta_loss_kls = torch.abs(ptrue_rep.KLs_min.to(loss_kls.device) - loss_kls)# if pars['delta_opt'] else loss_kls   
        
        
        #add KL for weighted constraints or randomly selected constraints to loss
        lkl_buf = delta_loss_kls if pars['delta_opt'] else loss_kls
        loss_kl = calculate_loss_kl(res, pars, lkl_buf, jm)     
        pars['omega_c'] = pars['omega']*pars['ds_fKL']    
        
    elif pars['use_Wu_loss'] == True:
        mbcov = ds.calculate_covariance(gbatch)
        loss_kl = pars['fcov']*torch.linalg.matrix_norm(ds.dcov-mbcov)
        pars['omega_c'] = pars['omega']*pars['fcov']
        loss_kls = torch.tensor(-1.)
        
    else:
        gloss = res.gloss0
        loss_kl = torch.tensor(0.)
        loss_kls = torch.tensor(0.)
        pars['omega_c'] = pars['omega']
        
    res.loss_kls = loss_kls
    return loss_kl

#calculate weighted loss term for constraints
def calculate_loss_kl(res, pars, lkl_buf, jm):
    if pars['Njkl'] > 0: #sample Njkl constraints according to weights
        loss_kl = 0
        for k in range(pars['Njkl']):            
            jkl, buf, res.akls, res.awkl = sample_and_update_weights(lkl_buf, res.awkl, res.akls)            
            loss_kl = loss_kl + buf/pars['Njkl'] #TODO: merge ds.fKL and omega
            res.ccount_ges[jm][jkl] += 1
    else: #use weighted sum of constraints        
        res.akls = lkl_buf - lkl_buf.min() + 0.1*(lkl_buf.max()-lkl_buf.min()) + 1e-4
        res.ccount_ges[jm] += 1
        
        if pars['constraint_weight_opt'] == 0:
            loss_kl = lkl_buf.mean()        
        elif pars['constraint_weight_opt'] == 1:
            loss_kl = (lkl_buf*res.akls/res.akls.sum()).sum()
        
    return loss_kl
    
    
#update learning rate via scheduler
def scheduler_step(dsched, gsched, freadme, itges):
    dsched.step()
    gsched.step()
    
    with open(freadme, 'a') as f:
        f.write('lr changed at it ='+str(itges)+'\n')





#######################
#######################
######################ä


#print loss values and add to results
def print_and_update(writer, res, pars, ptrue_rep, jm, epoch, itges, ti):
    if itges > 1:
        print("\r",
              'ep', epoch,
              'it', itges,
              'ld', round(np.array(res.losses[jm][0][-50:]).mean(),5),
              'ld2', round(np.array(res.losses[jm][1][-50:]).mean(),5),
              'lg', round(np.array(res.losses[jm][2][-50:]).mean(),5),
              'lg2', round(np.array(res.losses[jm][3][-50:]).mean(),5),
              'ti', round(time.time() - ti, 3),
              end="")
    
    res.losses[jm][0].append(res.dloss0.item())
    res.losses[jm][1].append(res.dloss_GP.item())
    res.losses[jm][2].append(res.gloss0.item())
    res.losses[jm][3].append(pars['omega']*res.loss_kl.item())
    
    
    ### tensorboard
    if res.itges%20 == 0:
        writer.add_scalar("loss: D", res.dloss, res.itges)
        writer.add_scalar("loss: G, ges", res.gloss, res.itges)
        writer.add_scalar("loss: G, standard", res.gloss0, res.itges)                    
        writer.add_scalar("loss: G, constraint", pars['omega_c']*res.loss_kl, res.itges)
        writer.add_scalar("loss: G, constraint (unscaled)", res.loss_kls.mean().detach().cpu(), res.itges) 
        if pars['use_pc_loss'] == True:
            res.gloss1_0 = torch.abs(ptrue_rep.KLs_min.to(res.loss_kls.device) - res.loss_kls).mean().detach().cpu()
            writer.add_scalar("loss: G, constraint (delta with optimal value)", res.gloss1_0, res.itges) 
    return res

#evaluate constraint fulfillment and performance metrics; plot intermediate results
def plot_and_evaluate(ds, res, writer, ptrue_rep, pfake_rep, pars, jm, dnet, gnet, device):
    teval = time.time()
    gnet.eval()
    plot = res.itges%res.plot_it == 0 and res.itges > 0
    plot2 = res.itges%(5*res.plot_it) == 0 and res.itges > 0
    
    if plot:
        #plot of fake samples
        gsamples = gnet(res.gsin)
        dscores = dnet((gsamples, res.gcval))
        fig = plot_samples(ds, (gsamples, dscores), res.ppathr_it + 'fake')        
        writer.add_figure("fake_samples", fig, res.itges)
        
        #plot of real samples
        dsamples = ds.data[:16,:]
        dscores0 = dnet((dsamples, res.gcval))
        plot_samples(ds, (dsamples, dscores0), res.ppathr_it + 'true')
        
    
    
    #generate bigger dataset to monitor statistics
    gges = generate_Ns(gnet, res.Neval, device, gnet.latent_dim).squeeze()
    cval_ges = ds.calculate_constraints(gges)
    pval_ges = ds.calculate_performance_metrics(gges)
    
    # if plot:
        # plot_summary(ds, gges, res.ppathr +'summary')
    
    #plot pmetric distribution
    pfig, pbuf = plot_pmetrics(ds, gges, res.ppathr, pars['match_opt'], plot=plot)
    if plot:
        writer.add_figure("pmetrics", pfig, res.itges)
        
    #plot constraint distributions (for selection)
    cfig, _ = plot_constraints(ds, gges, ptrue_rep, res.ppathr, pars['match_opt'], plot=plot, all_constraints=False)
    if plot:
        writer.add_figure("cmetrics", cfig, res.itges)
        
    #get histogram losses of constraints and metrics
    _, cbuf_abs = plot_constraints(ds, gges, ptrue_rep, res.ppathr, 'abs', plot=plot2, all_constraints=True)
    _, cbuf_KL = plot_constraints(ds, gges, ptrue_rep, res.ppathr, 'KL', plot=False, all_constraints=True)
    _, pbuf_abs = plot_pmetrics(ds, gges, res.ppathr, 'abs', plot=False)
    _, pbuf_KL = plot_pmetrics(ds, gges, res.ppathr, 'KL', plot=False)
    
    
    
    pfake_rep.update(cval_ges)
    pvec_true, log_pvec_fake, xvecs = get_p_representations(ptrue_rep, pfake_rep, 'xtrue')
    kls_ges = calculate_dist_loss(pvec_true, log_pvec_fake, xvecs, match_opt = pars['match_opt'])  
    delta_kls_ges = kls_ges.detach().cpu() - ptrue_rep.KLs_min      
    
    
    res.constraints_hist_abs_ges[jm].append(cbuf_abs)
    res.constraints_hist_KL_ges[jm].append(cbuf_KL)
    res.constraints_ges[jm].append(kls_ges.tolist())
    res.constraints_delta_ges[jm].append((delta_kls_ges).tolist())
    res.pmetrics_hist_abs_ges[jm].append(pbuf_abs)
    res.pmetrics_hist_KL_ges[jm].append(pbuf_KL)
    
    
    
    
    #eval metrics
    res.FID = abs(get_FID(ds.data, gges))
    res.cFID = abs(get_FID(ds.constraints, cval_ges))
    res.pFID = abs(get_FID(ds.pmetrics, pval_ges))
    writer.add_scalar('FID: samples', res.FID, res.itges)
    writer.add_scalar('FID: constraints', res.cFID, res.itges)
    writer.add_scalar('FID: pmetrics', res.pFID, res.itges)
    writer.add_scalar('constraints: values', kls_ges.mean().detach().cpu().item(), res.itges )
    writer.add_scalar('constraints: deltas', delta_kls_ges.mean().detach().cpu().item(), res.itges)
    writer.add_scalar('constraints: histogram - abs', np.array(cbuf_abs).mean().item(), res.itges)
    writer.add_scalar('constraints: histogram - KL', np.array(cbuf_KL).mean().item(), res.itges)
    writer.add_scalar('pmetrics: histogram - abs', np.array(pbuf_abs).mean().item(), res.itges)
    writer.add_scalar('pmetrics: histogram - KL', np.array(pbuf_KL).mean().item(), res.itges)
    
    
    
    cbuf = res.constraints_hist_abs_ges if pars['match_opt'] == abs else res.constraints_hist_KL_ges
    #plot loss curves
    if plot:
        plot_losses(res.losses[jm], res.ppathr + 'losses.pdf')
        plot_metric_trajectories(res.constraints_ges[jm], res.pmetrics_hist_abs_ges[jm], (ptrue_rep.cKLs_real, ptrue_rep.pKLs_hist_real), ds.constraint_names, ds.pmetric_names, res.eval_it, res.ppathr + 'KL_')
        #plot_metric_trajectories(res.constraints_ges0[jm], res.pmetrics_ges[jm], (ptrue_rep.cKLs_hist_real, ptrue_rep.pKLs_hist_real), ds.constraint_names, ds.pmetric_names, res.eval_it, res.ppathr + 'hist_')
        plot_metric_trajectories(cbuf[jm], res.pmetrics_hist_abs_ges[jm], (ptrue_rep.cKLs_hist_real, ptrue_rep.pKLs_hist_real), ds.constraint_names, ds.pmetric_names, res.eval_it, res.ppathr + 'hist_')
        for j in range(len(ds.pmetric_names)):
            writer.add_scalar('pmetric: ' + ds.pmetric_names[j], res.pmetrics_hist_abs_ges[jm][-1][j], res.itges)
        
    #plot constraint evaluation
    if plot:
        buf = np.array(res.constraints_ges[jm][-1])
        buf2 = np.array(res.constraints_delta_ges[jm][-1])
        buf3 = res.ccount_ges[jm] if pars['Njkl'] > 0 else (res.akls/res.akls.sum()).detach().cpu()
        cfig = plot_constraint_fulfillment(buf, buf2,  buf3, res.ppathr)
        writer.add_figure("constraint violations", cfig, res.itges)
    
    gnet.train()
    res.ieval += 1
    
