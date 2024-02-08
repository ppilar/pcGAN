# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pcGAN.utils import *
from pcGAN.utils_train import initialize_ds, initialize_ptrue_rep
from pcGAN.representations.utils_representations import sample_mb
from pcGAN.representations.rKDE import rKDE
from pcGAN.axplots import axplot_recent_mbs, axplot_hist

if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=5) #set random seed



def axplot_pfake_pvec(ax, ptrue_rep, cmb, x, jc, fsig=False, label= ''):
    pfake = rKDE(ptrue_rep, fsig)
    pfake.update(cmb)
    p = pfake.get_pvec(x, jc)
    ax.plot(x, p, label=label)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #select device
fpath0 =  '../results/datasets/' 

pars = dict()
pars['ds_opt'] = 1
pars['ptrue_rep'] = 'KDE'
pars['load_ds'] = 1
pars['load_ptrue_rep'] = 1
pars['match_opt'] = 'KL'
pars['bs'] = 64
ds = initialize_ds(pars, device, fpath0) #initialize dataset
ptrue_rep = initialize_ptrue_rep(ds, pars, '', fpath0)


jc = 1
cdata = ds.constraints[:,jc].cpu()
sig0 = ((ds.constraints.max(0)[0] - ds.constraints.min(0)[0])/200)
fsig0 = ptrue_rep.cstds/sig0
sig_best = ptrue_rep.cstds/ptrue_rep.fsig_best
sig2 = ((ds.constraints.max(0)[0] - ds.constraints.min(0)[0])/10)
fsig2 = ptrue_rep.cstds/sig2
x = ptrue_rep.xvecs[jc].cpu()[20:300]
p = ptrue_rep.get_pvec(x, jc)



fig, axs = plt.subplots(1,4, figsize=(18.2,4))
axs[0].plot(x, p)
axs[0].set_title(r'$\tilde p_{\rm true}$', fontsize=14)
#axplot_hist(axs[0], x, -1, (x.min().item(), x.max().item()))

mb, cmb = sample_mb(ds, pars['bs'])
axplot_pfake_pvec(axs[1], ptrue_rep, cmb, x, jc, fsig0, label=r'$\sigma$ = %.1f'%(sig0[jc].item()))
axplot_pfake_pvec(axs[1], ptrue_rep, cmb, x, jc, label=r'$\sigma$ = %.1f'%(sig_best[jc].item()))
axplot_pfake_pvec(axs[1], ptrue_rep, cmb, x, jc, fsig2, label=r'$\sigma$ = %.1f'%(sig2[jc].item()))
axs[1].legend()
axs[1].set_title(r'$\tilde p_{\rm gen}$ (bs = 64)', fontsize=14)


match_opt = 'KL'
bsvec = [32,64,256]
for bs in bsvec:
    ptrue_rep.check_match_bs(ds, match_opt, bs)
    mb, cmb = sample_mb(ds, bs)
    axplot_pfake_pvec(axs[2],ptrue_rep, cmb, x, jc)
axs[2].legend(['bs='+str(bs) for bs in bsvec])
axs[2].set_title(r'$\tilde p_{\rm gen}$ (optimal $\sigma$)', fontsize=14)
    
for ax in axs:
    ax.set_ylim([-0.0005, 0.05])
    ax.hist(cdata, bins=100, density=True, color='grey', alpha = 0.3)
    ax.set_xlim([x.min(),x.max()])
    ax.set_xlabel(r'$z_s$')
    

bs = 32
ptrue_rep.check_match_bs(ds, match_opt, bs)
pfake = rKDE(ptrue_rep)
for j in range(50):
    mb, cmb = sample_mb(ds, bs)
    pfake.update(cmb)
    pfake.pvec(x)
    
Nh = 10
axplot_recent_mbs(axs[3], Nh, pfake, jc, x, cmb_label=False)
axs[3].plot(x, pfake.exponential_decay()[jc].cpu(), label='bs='+str(bs)+' (with history)', color='black')
axs[3].legend()
axs[3].set_title(r'$\tilde p_{\rm gen}^i$', fontsize=14)

plt.savefig('../plots/representations.pdf', bbox_inches='tight')