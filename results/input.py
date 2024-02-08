#define settings
if not 'pars' in locals(): pars = dict() #dictionary that contains training parameters and settings
if not 'comment' in locals(): comment = '' #dictionary that contains training parameters and settings

init_par(pars, 'model_vec', [1])  
#0 ... (...)GAN, 1 ... GAN + KL, 2 ... KL, 3 ... covariance (Wu et al.), 4 ... WGAN-GP, 5 ... SNGAN
#6 ... WGAN-GP + KL, 7 ... SNGAN + KL
init_par(pars, 'ds_opt', 1) #0 ... sum of numbers; 1 ... wave forms; 2 ... CAMELS; 3 ... IceCube
init_par(pars, 'load_ds', 1)
init_par(pars, 'load_ptrue_rep', 1)
        

init_par(pars, 'itmax', 100000) #150000
init_par(pars, 'Nrun', 1)

init_par(pars, 'bs', 128) #batch size
init_par(pars, 'omega', 5) #weighting factor




        
