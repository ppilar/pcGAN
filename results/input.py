#define settings
if not 'ds_opt' in locals(): ds_opt = 1      #0 ... sum of numbers; 1 ... wave forms; 2 ... CAMELS; 3 ... IceCube
itmax = 150000
mode_vec = [1,0,3]   #0 ... (...)GAN, 1 ... GAN + KL, 2 ... KL, 3 ... covariance (Wu et al.)
load_ds_cebm = 0

#the below parameters can be iterated over from master.py
if not 'GAN_opt' in locals(): GAN_opt = 1       #0 ... GAN, 1 ... WGAN, 2 ... WGAN-GP
if not 'bs' in locals(): bs = 256 #batch size
if not 'Nd' in locals(): Nd = 1   #discriminator updates per iteration
if not 'Njkl' in locals(): Njkl = 3   #discriminator updates per iteration
