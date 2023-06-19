# -*- coding: utf-8 -*-

import sys
import time
import torch
import numpy as np
from .Nets import cNet
from .utils import update_wvec

class cEBM():
    #initialize cEBM
    def __init__(self, ds_name, cdata, constraint_names, device, Nit=20000, cut_tails = True):
        self.ds_name = ds_name
        self.device = device        
        self.cut_tails = cut_tails
        self.constraint_names = constraint_names
        self.extract_parameters(cdata)
        self.initialize_model(cdata)
        tx = time.time()
        self.initialize_xvecs()
        print('tx:%.3f' %(time.time() - tx))
        self.train(Nit)
    
    #forward through net, if called
    def __call__(self, x):
        return self.net_ebm(x)
        
    #initialize network parameters and miscallaneous values
    def initialize_model(self, cdata):
        
        ebms = []
        Jebms = []
        
        self.NJ = np.minimum(10,self.Nc) #number of classes considered in each iteration
        self.lr = 5e-3
        self.batch_size = 1024
        
        self.dr = 0.2
        self.Uvec = [40]*3
        self.net_ebm = cNet(Uvec = self.Uvec, iovec = [1,1], fdrop=self.dr, bounds = self.ebm_bounds, device = self.device).to(self.device)
        
        self.optimizer_ebm = torch.optim.Adam(self.net_ebm.parameters(), lr=self.lr)
        self.scheduler_ebm = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_ebm, 0.5)
        self.ebm_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            cdata), batch_size = self.batch_size, num_workers=0,  shuffle=True)
        
        #        
        self.aJ = np.ones(self.Nc) #keep track of nlls for individual classes; used for weighted sampling
        self.Jebm = []
        self.tiges = []
        self.it_ebm = 0
        self.Nit_max = 0
        
        #keep track of fsigs
        self.abs = [] #batch sizes
        self.afsig_best = [] #best values of fsig given batch size
        self.aKLs_min = [] #min KL achieved given fsig        
        self.acKLs_real = []
        self.acKLs_hist_real = []
        self.apKLs_hist_real = []
        
        
        #current values
        self.bs = -1
        self.fsig_best = -1
        self.KLs_min = -1
        self.cKLs_real = -1
        self.cKLs_hist_real = -1
        self.pKLs_hist_real = -1
        
    #extract parameters from constraint data
    def extract_parameters(self, cdata):
        self.Nc = cdata.shape[1]
        self.cmins = cdata.min(0)[0]
        self.cmaxs = cdata.max(0)[0]
        self.cstds = cdata.std(0)
        self.ccats = torch.linspace(0,self.Nc-1,self.Nc).int()
        
        self.ebm_out_u = 5 
        self.ages = self.ebm_out_u*torch.ones(self.Nc)        
        self.ebm_bounds = (self.cmins - 0.1*self.cstds, self.cmaxs + 0.1*self.cstds, 0, self.ages)
        
    #train the model
    def train(self, Nit=5000):
        self.train_iterate(Nit)        
        with torch.no_grad():
            self.normalize()

    #update the weights during training
    def update_wvec(self, aJ):
        wexp = 1
        buf = (aJ - aJ.min() + (aJ.max() - aJ.min())*0.1 + 1e-4)**wexp
        wvec = buf/buf.sum()
        return wvec
        
    #training loop
    def train_iterate(self, Nit = 5000):
        self.Nit_max += Nit
        self.net_ebm.train()        
        
        wvec = self.update_wvec(self.aJ)
        
        while(True):
            for i, batch in enumerate(self.ebm_loader):
                t0 = time.time()
                
                J = 0
                buf = batch[0][:,:]
                
                if self.ds_name == 'IceCube' and self.constraint_names == ['max','min']: #add small noise to singular PDF of min/max values in IceCube constraints
                    buf = add_mm_noise(buf, self.device)
                    
                
                Js = self.get_mean_NLL(self.net_ebm, buf, range(0,self.Nc))                
                for j in range(self.NJ):
                    jcx = np.random.choice(range(self.Nc), 1, p=wvec).item() #weight lower frequencies more strongly
                    x_batch = batch[0][:,jcx]
                        
                    Jbuf = Js[jcx]
                    self.aJ[jcx] = (self.aJ[jcx] + Jbuf.item())/2 
                    J += Jbuf
                    wvec = self.update_wvec(self.aJ)
                J = J/self.NJ
                    
                self.optimizer_ebm.zero_grad()
                J.backward()
                self.optimizer_ebm.step()
                self.Jebm.append(J.item())
                
                self.it_ebm += 1
                if self.it_ebm%6000 == 0 and self.it_ebm < 15000:
                    self.scheduler_ebm.step()
                
                if self.it_ebm == self.Nit_max:
                    break
                
                self.tiges.append(time.time() - t0)
                
                if self.it_ebm%100 == 0 and self.it_ebm >= 100:
                    Jebm_mean = np.array(self.Jebm[-100:]).mean()
                    print("\r", 'it=' + str(self.it_ebm),'Jebm=%.3f' % Jebm_mean, 'tit=%.4f' % np.array(self.tiges[-100:]).mean(), end="")
            
            if self.it_ebm == self.Nit_max:
                break
            
        self.net_ebm.eval()
        
    #calculate mean negative log-likelihood 
    def get_mean_NLL(self, net, res, jcx):
        Nres = res.shape[0]
        
        clabels = self.cvalues_l[jcx,0].repeat(Nres,1)
        
        rvec = self.xvecs_l[jcx]
        clabels_rvec = self.cvalues_l[jcx]
        
        
        buf = net((rvec.flatten().unsqueeze(1).float(), clabels_rvec.flatten())).squeeze().reshape(len(jcx),-1)
        mbuf = buf.max(1)[0].unsqueeze(1)
        buf_Z = torch.exp(buf - mbuf)
        buf_res = net((res.flatten().unsqueeze(1), clabels.flatten())).squeeze().reshape(Nres,-1)

            
        Z = torch.trapezoid(buf_Z, rvec, dim=1)
        J = -torch.sum(buf_res,0) + Nres*torch.log(Z) + Nres*mbuf.squeeze()
        J2 = J/Nres
        return J2
        
    #initilize x-vectors on which the PDF will be calculated
    def initialize_xvecs(self):
        xmins = self.cmins
        xmaxs = self.cmaxs
        jcx = self.ccats
        Nx = 201
        
        xvecs = torch.linspace(xmins[0].item(), xmaxs[0].item(), Nx).unsqueeze(0)
        cvalues = jcx[0]*torch.ones(Nx, device=self.device).int().unsqueeze(0)
        for j in range(1,self.Nc):
            buf = torch.linspace(xmins[j].item(), xmaxs[j].item(), Nx).unsqueeze(0)
            xvecs = torch.cat((xvecs, buf), 0)
            buf2 = jcx[j]*torch.ones(Nx, device=self.device).int().unsqueeze(0)
            cvalues = torch.cat((cvalues, buf2), 0)
        self.xvecs = xvecs.to(self.device)
        self.cvalues = cvalues.to(self.device)
        
        xmins_l = self.cmins - self.cstds/3
        xmaxs_l = self.cmaxs + self.cstds/3
        Nx_l = 501
        xvecs_l = torch.linspace(xmins_l[0].item(), xmaxs_l[0].item(), Nx_l).unsqueeze(0)
        cvalues_l = jcx[0]*torch.ones(Nx_l, device=self.device).int().unsqueeze(0)
        for j in range(1,self.Nc):
            buf = torch.linspace(xmins_l[j].item(), xmaxs_l[j].item(), Nx_l).unsqueeze(0)
            xvecs_l = torch.cat((xvecs_l, buf), 0)
            buf2 = jcx[j]*torch.ones(Nx_l, device=self.device).int().unsqueeze(0)
            cvalues_l = torch.cat((cvalues_l, buf2), 0)
        self.xvecs_l = xvecs_l.to(self.device)
        self.cvalues_l = cvalues_l.to(self.device)
        
    #normalize the PDFs
    def normalize(self):
        self.p0vec = self.net_ebm((self.xvecs.flatten().unsqueeze(1).float(), self.cvalues.flatten())).squeeze().reshape(self.Nc,-1)
        mbufs = self.p0vec.max(1)[0].unsqueeze(1) #normalize to lie within reasonable range
        self.p0vec = self.p0vec - mbufs
        
        
        self.norm_p0 = torch.trapz(torch.exp(self.p0vec), self.xvecs, dim=1).unsqueeze(1)
        self.pvec = (torch.exp(self.p0vec)/self.norm_p0).detach()
        
        #cut off tails
        self.poff = (self.pvec.max(1)[0]/20).unsqueeze(1)
        self.pvec_cut = torch.maximum(torch.tensor(0), self.pvec - self.poff)
        self.pvec_cut = self.pvec_cut/(torch.trapz(self.pvec_cut, self.xvecs, dim=1).unsqueeze(1))

    #check if batch size bs has already been considered; if not add empty entries to arrays
    def check_bs(self, bs):
        if bs not in self.abs:
            bsnew = True
            self.abs.append(bs)
            self.aKLs_min.append([])
            self.afsig_best.append([])
            self.acKLs_real.append([])
            self.acKLs_hist_real.append([])
            self.apKLs_hist_real.append([])
            ibs = len(self.abs)-1
        else:
            ibs = np.where(np.array(self.abs)==bs)[0][0]        
        return ibs
    
    #set currently used batch size
    def set_batch_size(self, bs):
        ibs = self.check_bs(bs)
        self.bs = self.abs[ibs]
        self.fsig_best = self.afsig_best[ibs]
        self.KLs_min = self.aKLs_min[ibs]
        self.cKLs_real = self.acKLs_real[ibs]
        self.cKLs_hist_real = self.acKLs_hist_real[ibs]
        self.pKLs_hist_real = self.apKLs_hist_real[ibs]
    
    #calculate KL and optimal values of fsig for batches of size bs for real data
    def calculate_real_data_metrics(self, ds, bs, Neval):
        self.find_optimal_fsig(ds, bs)
        self.find_optimal_fsig(ds, Neval)
        self.calculate_real_data_KL(ds, bs, Neval)
        self.calculate_real_data_KL_hist(ds, bs, Neval)
        self.set_batch_size(bs)
        
    #determine optimal values fsig given batch size bs
    def find_optimal_fsig(self, ds, bs):
        
        ibs = self.check_bs(bs)
        
        print('finding optimal fsig for bs='+str(bs))
        if bs > 10000:
            Navg = 5 #avoid too long runtime for very big batch sizes (such as when using Neval); for such big batches, the variance between batches is also expected to be smaller
        else:
            Navg = 50
        Nsig = 200
        Nc = ds.constraints.shape[1]
        
        fsig_vec = np.logspace(-1,2,Nsig)
        KLs = torch.zeros(Nc, Nsig, Navg)
        for jN in range(Navg):
            sys.stdout.write('\r'+str(jN))
            sys.stdout.flush()
            
            mb, cmb = sample_mb(ds, bs)
            for jsig, fsig in enumerate(fsig_vec):
                KLs[:, jsig, jN] = calculate_multiple_KL(self, cmb[:,:], fsig = fsig, cut_tails=self.cut_tails).detach().cpu()
                
        KLs_min, iKLs_min = KLs.mean(-1).min(-1)
        fsig_best = torch.tensor(fsig_vec[iKLs_min]).float().to(ds.device)
        
        self.aKLs_min[ibs] == KLs_min
        self.afsig_best[ibs] = fsig_best
            
        
    #determine the KL divergence between real data and cEBM PDF
    #the model has been trained on batch size bs but is evaluated on a generated dataset of size Neval
    def calculate_real_data_KL(self, ds, bs, Neval):        
        print('calculating real data KL')
        
        ibs0 = self.check_bs(bs) #index corresponding to batch of size bs
        ibs = self.check_bs(Neval) #index corresponding to batch of size Neval
        
        Navg = 50 #number of datasets to average over
        Nc = ds.constraints.shape[1]
        fsig_vec = self.afsig_best[ibs]
        if fsig_vec.ndim == 0:
            fsig_vec = fsig_vec.unsqueeze(0)
        
        KLs = torch.zeros(Nc, Navg)
        for jN in range(Navg):
            sys.stdout.write('\r'+str(jN))
            sys.stdout.flush()
            
            mb, cmb = sample_mb(ds, Neval)
            if type(fsig_vec) == int:
                KLs[:,jN] = calculate_multiple_KL(self, cmb, fsig=fsig_vec, cut_tails=self.cut_tails).detach().cpu()
            else:
                for jsig, fsig in enumerate(fsig_vec):
                    KLs[jsig,jN] = calculate_multiple_KL(self, cmb[:,jsig].unsqueeze(1), jsig, fsig=fsig.unsqueeze(0), cut_tails=self.cut_tails).detach().cpu()
        
        
        self.acKLs_real[ibs0] = KLs.mean(-1)
        #return KLs.mean(-1)
        
    #calculate KL divergence between histograms, instead of between cEBM & mixture of Gaussians
    def calculate_real_data_KL_hist(self, ds, bs, Neval):    
        print('calculating real data KL hist')
        
        ibs = self.check_bs(bs) #index corresponding to batch of size bs
        Navg = 50
        
        cKLs_hist = torch.zeros(ds.Nebm, Navg)
        pKLs_hist = torch.zeros(ds.Np, Navg)
        
        for jN in range(Navg):
            sys.stdout.write('\r'+str(jN))
            sys.stdout.flush()
            
            mb, _ = sample_mb(ds, Neval)
            cKLs_hist[:,jN] = torch.tensor(ds.plot_constraints(mb, self, '', plot=False))
            pKLs_hist[:,jN] = torch.tensor(ds.plot_pmetrics(mb, '', plot=False))
            
        self.acKLs_hist_real[ibs] = cKLs_hist.mean(-1)
        self.apKLs_hist_real[ibs] = pKLs_hist.mean(-1)
        
        #return cKLs_hist.mean(-1), pKLs_hist.mean(-1)
        
#calculate KL divergence between PDF from cEBM and mixture of Gaussians from minibatch
def calculate_multiple_KL(net_ebm, xnet, jcx = False, fsig = 7, cut_tails=True):
    device = xnet.device
    Ns = xnet.shape[0]
    
    if type(jcx) == bool:
        Nc0 = 0
        Nc = xnet.shape[1]
    else:
        Nc0 = jcx
        Nc = jcx+1
    
    sigs = (net_ebm.cstds/fsig)[Nc0:Nc]
    xvecs = net_ebm.xvecs[Nc0:Nc,:]
    if cut_tails:
        pvec = net_ebm.pvec_cut[Nc0:Nc,:]
    else:
        pvec = net_ebm.pvec[Nc0:Nc,:]    
        
    Nx = pvec.shape[1]
    
    xmeans = xnet.unsqueeze(2)
    xsigs = sigs.unsqueeze(0).unsqueeze(2)
    xvals = xvecs.unsqueeze(0)
    
    
    log_prob = log_prob_gaussian(xvals, xmeans, xsigs) 
    bmax = log_prob.max(0)[0]
    buf = log_prob - bmax.unsqueeze(0)
    lq = torch.exp(buf).sum(0)/Ns
    
    rvec = pvec*(torch.log((pvec+1e-9)/(lq + 1e-9)) - bmax)
    rges = torch.trapz(rvec, xvecs, dim=1)
    return rges

#calculate log probability of Gaussian
def log_prob_gaussian(x, mean, std):
    log_prob = -0.5 * torch.log(2 * torch.tensor(np.pi).float()) - torch.log(std) - 0.5 * ((x - mean) / std)**2
    return log_prob

#add small amount of noise to min/max values of IceCube data to avoid singular PDF
def add_mm_noise(buf, device):
    abuf = torch.rand(buf.shape[0]).to(device)       
    for k in range(2):
        ar = -0.01*abuf if k == 0 else 0.01*abuf
        buf[:,k] = buf[:,k] + ar
    return buf

#sample minibatch
def sample_mb(ds, bs=64):
    perm = torch.randperm(ds.data.size(0))
    idx = perm[:bs]
    return ds.data[idx], ds.constraints[idx]