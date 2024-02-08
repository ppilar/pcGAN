# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils import log_prob_gaussian, write_and_flush, calculate_dist_loss
from ..plots import plot_constraints, plot_pmetrics
from .utils_representations import *

class Representation():
    def __init__(self, include_history, Nh=10, fforget=0.9):
        self.include_history = include_history #0 ... only current minibatch; 1 ... exponential decay; 2 ... weighted average of N recent minibatches
        self.Nh = Nh #number of minibatches to take into account for weighted average
        self.history = [] #collection of recent minibatches
        self.hist_exp = 0 #summary of preceding minibatches via exponential decay
        self.fforget = fforget  #forgetting factor for exponential decay
        self.hfacs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #weighting factors for weighted average
        self.hfacs_exp = self.fforget**np.linspace(9,0,10)
        
    def pvec(self, xvecs='xtrue'):        
        pvec = self.get_pvec(xvecs)
        
        self.add_to_history(pvec)
        if self.include_history == 1:
            pvec = self.exponential_decay()            
        if self.include_history == 2:
            pvec = self.weighted_average()
        
        self.detach_history(pvec)                   
        return pvec
    
    def add_to_history(self, pvec):   
        self.history.append(pvec)
        if len(self.history) > self.Nh:
            self.history.pop(0)
        
    def exponential_decay(self):
        pvec = self.history[-1]
        if type(pvec) == torch.Tensor:
            pvec = (1-self.fforget)*pvec + self.fforget*self.hist_exp
        return pvec
    
    def weighted_average(self):
        pvec = self.history[-1]
        if len(self.history) == self.Nh:
            for i, pvec_h in enumerate(self.history[:-1]):                
                pvec = pvec + pvec_h*self.hfacs[-self.Nh+i]
            norm = torch.tensor(self.hfacs[-self.Nh:]).sum()
            pvec = pvec/norm
        return pvec
    
    def detach_history(self, pvec):
        self.history[-1] = self.history[-1].detach()
        self.hist_exp = pvec.detach()
    
    # def plot_history():
    #     plt.figure()
    #     plt.plot(self.history[0][i,:].detach().cpu())       
    #     plt.plot(self.history[1][i,:].detach().cpu()) 
    #     plt.plot(pvec[i,:].detach().cpu(), color = 'k')
    #     plt.show()


    ##############################
    ############################## 
    def initialize_real_data_rep(self, pars, cdata):
        self.Neval = pars['Neval']
        self.initialize_arrays()
        self.extract_parameters(cdata)
        self.initialize_xvecs()
        
    
    
    def extract_parameters(self, cdata):
        self.Nc = cdata.shape[1]
        self.cmins = cdata.min(0)[0]
        self.cmaxs = cdata.max(0)[0]
        self.cq01 = torch.quantile(cdata, 0.01, dim=0)
        self.cq99 = torch.quantile(cdata, 0.99, dim=0)
        off = (self.cq99 - self.cq01)*0.05
        self.crange_min = self.cq01-off
        self.crange_max = self.cq99+off
        
        self.cstds = cdata.std(0)
        self.ccats = torch.linspace(0,self.Nc-1,self.Nc).int()
        
        self.device = cdata.device
    
    def get_crange(self, jc, crange_opt = ''):
        if crange_opt == '':
            return (self.crange_min[jc].item(), self.crange_max[jc].item())            
        if crange_opt == 'minmax':
            return (self.cmins[jc].item(), self.cmaxs[jc].item())
        if crange_opt == 'plot':
            if jc == 12:
                return (self.crange_min[jc].item(), 20)
            if jc in [15, 17]:                
                return (self.crange_min[jc].item(), self.cmaxs[jc].item()/6)
            
            return self.get_crange(jc)
            
            
    def determine_xvecs(self, l_opt = 0):
        if l_opt == 0:
            xmins = self.cmins
            xmaxs = self.cmaxs
            Nx = 201
        elif l_opt == 1:
            xmins = self.cmins - self.cstds/3
            xmaxs = self.cmaxs + self.cstds/3
            Nx = 501
        jcx = self.ccats
        
        xvecs = torch.zeros(self.Nc, Nx)
        cvalues = torch.zeros(self.Nc, Nx)
        for j in range(self.Nc):
            xvecs[j] = torch.linspace(xmins[j].item(), xmaxs[j].item(), Nx)
            cvalues[j] = jcx[j]*torch.ones(Nx).int()
        
        return xvecs.to(self.device), cvalues.to(self.device)
    
    #initialize x-vectors on which the PDF will be calculated
    def initialize_xvecs(self): 
        #self.xvecs_l, self.cvalues_l = self.determine_xvecs( l_opt = 0)
        self.xvecs, self.cvalues = self.determine_xvecs( l_opt = 1)
          
    
    
    
    ###############################
    ###############################
    
    def initialize_arrays(self):
        #keep track of fsigs; used for matching the distributions
        self.Navg = 50 # how many batches to consider for determining fsig_best
        self.amatch_opt = [] #different match options considered
        self.abs = [] #batch sizes
        self.afsig_best = [] #best values of fsig given batch size
        self.aKLs_min = [] #min KL achieved given fsig        
        self.acKLs_real = []
        self.acKLs_hist_real = []
        self.apKLs_hist_real = []
        self.lists_match_par = [self.abs, self.afsig_best, self.aKLs_min, self.acKLs_real, self.acKLs_hist_real, self.apKLs_hist_real]
        
        
        #current values
        self.bs = -1
        self.fsig_best = -1
        self.KLs_min = -1
        self.cKLs_real = -1
        self.cKLs_hist_real = -1
        self.pKLs_hist_real = -1
    
    
    def check_match_bs(self, ds, match_opt, bs):
        imo = self.check_match(match_opt)
        if bs not in self.abs[imo]:
            self.calculate_real_data_metrics(ds, bs, match_opt, Neval=self.Neval)
        else:
            self.set_batch_size(match_opt, bs)

        
    #check if match_opt has already been considered; if not append empty lists
    def check_match(self, match_opt):
        if match_opt not in self.amatch_opt:
            self.amatch_opt.append(match_opt)            
            for l in self.lists_match_par:
                l.append([])
            imo = len(self.amatch_opt)-1
        else:
            imo = np.where(np.array(self.amatch_opt)==match_opt)[0][0]
        return imo
            

    #check if batch size bs has already been considered; if not add empty entries to arrays
    def check_bs(self, imo, bs):
        if bs not in self.abs[imo]:
            self.abs[imo].append(bs)            
            for l in self.lists_match_par[1:]:
                l[imo].append([])
            ibs = len(self.abs[imo])-1
        else:
            ibs = np.where(np.array(self.abs[imo])==bs)[0][0]        
        return ibs
    
    #set currently used batch size
    def set_batch_size(self, match_opt, bs):
        imo = self.check_match(match_opt)
        ibs = self.check_bs(imo, bs)
        self.bs = self.abs[imo][ibs]
        self.fsig_best = self.afsig_best[imo][ibs]
        self.KLs_min = self.aKLs_min[imo][ibs]
        #self.cKLs_real = self.acKLs_real[imo][ibs]
        self.cKLs_hist_real = self.acKLs_hist_real[imo][ibs]
        self.pKLs_hist_real = self.apKLs_hist_real[imo][ibs]
    
    #calculate KL and optimal values of fsig for batches of size bs for real data
    def calculate_real_data_metrics(self, ds, bs, match_opt, Neval):
        self.find_optimal_fsig(ds, bs, match_opt)
        self.find_optimal_fsig(ds, Neval, match_opt)
        self.calculate_real_data_KL(ds, bs, Neval, match_opt)
        self.calculate_real_data_KL_hist(ds, bs, Neval, match_opt)
        self.set_batch_size(match_opt, bs)
        
    #determine optimal values fsig given batch size bs
    def find_optimal_fsig(self, ds, bs, match_opt): #TODO: adjust to calculate for different metrics; different metrics in same cebm?        
        from .rKDE import rKDE
        print('\nfinding optimal fsig for bs='+str(bs)+' with match_opt='+match_opt)
        
        imo = self.check_match(match_opt)
        ibs = self.check_bs(imo, bs) 
        Navg = 1 if bs > 3000 else self.Navg  #avoid too long runtime for very big batch sizes (such as when using Neval); for such big batches, the variance between batches is also expected to be smaller
       
        
        
        KLs, fsig_vec = calculate_KLs_for_fsig(ds, self, match_opt, bs, Nsig = 200, Navg = Navg)                
        KLs_min, iKLs_min = KLs.mean(-1).min(-1)
        fsig_best = torch.tensor(fsig_vec[iKLs_min]).float().to(ds.device)
        
        self.aKLs_min[imo][ibs] = KLs_min
        self.afsig_best[imo][ibs] = fsig_best
            
    # determine the KL divergence between real data and cEBM PDF
    # the model has been trained on batch size bs but is evaluated on a generated dataset of size Neval
    def calculate_real_data_KL(self, ds, bs, Neval, match_opt):
        imo = self.check_match(match_opt)
        ibs0 = self.check_bs(imo, bs) #index corresponding to batch of size bs
        ibs = self.check_bs(imo, Neval) #index corresponding to batch of size Neval
        
        self.acKLs_real[imo][ibs0] = self.aKLs_min[imo][ibs]
        
        
    #calculate KL divergence between histograms, instead of between cEBM & mixture of Gaussians
    def calculate_real_data_KL_hist(self, ds, bs, Neval, match_opt):    #TODO: needs to be revised/or removed
        print('\ncalculating real data KL hist')
        from .rKDE import rKDE
        
        imo = self.check_match(match_opt) #index corresponding to chosen match option
        ibs = self.check_bs(imo, bs) #index corresponding to batch of size bs
        Navg = 1 if bs > 3000 else self.Navg
        
        
        cKLs_hist = torch.zeros(ds.Nebm, Navg)
        pKLs_hist = torch.zeros(ds.Np, Navg)        
        for jN in range(Navg):          
            write_and_flush(str(jN))
            
            mb, _ = sample_mb(ds, Neval)
            cKLs_hist[:,jN] = torch.tensor(plot_constraints(ds, mb, self, '', match_opt, plot=False)[1]) #TODO: can it be disentangled from plotting?
            pKLs_hist[:,jN] = torch.tensor(plot_pmetrics(ds, mb, '', match_opt, plot=False)[1])
            
        self.acKLs_hist_real[imo][ibs] = cKLs_hist.mean(-1)
        self.apKLs_hist_real[imo][ibs] = pKLs_hist.mean(-1)
    
