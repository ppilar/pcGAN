# -*- coding: utf-8 -*-


from .datasets import dataset
import matplotlib.pyplot as plt


#Temperature maps from the CAMELS project
class CAMELS(dataset):
    def __init__(self, device='cpu', pars=()):
        
        self.set_power_spectrum_pars()
        super().__init__(-1, device, pars)
        self.clamp = 2e-2
        self.latent_dim = 100
        self.D_batches_together = False
        
        
        self.lr = 1e-4
        
        #prefactors for added loss terms
        self.fKL = 0.01
        self.fKLG = 0.0025 
        self.fm = 0.5
        self.fcov = 0.001
        
        self.ds_name = 'Tmaps'
        self.sample_ptype = plt.imshow
        
        self.s6_figsize = (9,6)
        self.s16_figsize = (12,12)
        
        
        self.cinds_selection = [0,1,2,5,10,20,30]
        
        
    def generate_data(self, Ns, jmax = 1):
        dpath = '../data/'
        fname = 'Tmaps.pickle'
        with open(dpath + fname,'rb') as f:
            Cdata = torch.tensor(pickle.load(f), device=self.device)

        self.Ns = Cdata.shape[0]
        
        return Cdata
    
    def init_cp_names(self):
        self.constraint_names = ['ps [' + str(j) + ']' for j in range(32)]
        #athr = [-0.5, 0.0, 0.5]
        athr = [0.8, 0.9, 0.95]
        self.pmetric_names = ['Mk 0 ('+str(a)+')' for a in athr ]
        self.pmetric_names.extend(['Mk 1 ('+str(a)+')' for a in athr ])
        self.pmetric_names.extend(['Mk 2 ('+str(a)+')' for a in athr ])
        #self.pmetric_names = ['Tpdf']
        
    def calculate_metrics(self, data, mlist):
        Ns = data.shape[0]
        Nm = len(mlist)
        
        
        #fftges, fftstats = get_fft(data)
        power_spectrum = self.calculate_power_spectrum(data).to(self.device)
        metrics = torch.zeros(Ns, Nm).to(self.device)
        for jm, m in enumerate(mlist):
            if m == 'Tpdf':
                metrics[:,jm] = self.calculate_Tges(data)[np.random.randint(0,data.numel(),Ns)]
            if m[:2] == 'ps':
                i = int(m[4:-1])
                metrics[:,jm] = power_spectrum[:,i]
            if m[:2] == 'Mk':
                i = int(m[3])
                tbuf = m[6:-1]
                thr = float(tbuf)
                
                metrics[:,jm] = get_Minkowski(data, i, thr)

        
        return metrics
    
    def get_optimizers(self, gnet, dnet):
        gopt = torch.optim.Adam(gnet.parameters(), lr=self.lr, betas = (0., 0.9))
        dopt = torch.optim.Adam(dnet.parameters(), lr=self.lr, betas = (0., 0.9))
        return gopt, dopt
    
    def get_GAN_nets(self):
        hidden_dim = 64 #?
        gnet = Generator_64(self.latent_dim, hidden_dim).to(self.device)
        dnet = Discriminator_64(64).to(self.device)
        gnet.latent_dim = self.latent_dim
        return gnet, dnet
    
    def calculate_Tges(self, data):
        Tges = data.reshape(-1)
        return Tges
    
    def set_power_spectrum_pars(self):
        npix = 64
        #adapted from https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
        self.kbins = torch.arange(0.5, npix//2+1, 1.)
        kvals = 0.5*(self.kbins[1:] + self.kbins[:-1])
        kfreq = torch.fft.fftfreq(npix)*npix
        kfreq2D = torch.meshgrid(kfreq, kfreq)
        knrm = torch.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        knrm = knrm.flatten()
        #a0, b0 = torch.histogram(knrm, bins=kbins) #get bin counts
        inds, self.arrs, self.Nind = get_knrm_indices(knrm, self.kbins)
        
        
    def calculate_power_spectrum(self, data):
        data = data.cpu() #does not work with cuda?
        Ns = data.shape[0]
        npix = data.shape[-1]

        fft_ges = torch.fft.fft2(data)
        fft_amplitudes_ges = (torch.abs(fft_ges)**2).flatten(start_dim=1)
        ps0_ges = torch.sparse.mm(self.arrs.float(),fft_amplitudes_ges.T).T/self.Nind
        ps_ges = ps0_ges[:,1:-1]* torch.pi * (self.kbins[1:]**2 - self.kbins[:-1]**2)
        
        return ps_ges
        
    #augment data by randomly rotating or flipping Tmaps
    def augment_batch(self, batch, dim_offset = 0, opt = -1):
        if opt == -1:
            opt = np.random.randint(8)
            
        dims = (1+dim_offset,2+dim_offset)
        dims_flip = (0+dim_offset,1+dim_offset)
        
        if opt == 0:
            batch = batch
        elif opt == 1:
            batch = torch.rot90(batch,1,dims=dims)
        elif opt == 2:
            batch = torch.rot90(batch,2,dims=dims)
        elif opt == 3:
            batch = torch.rot90(batch,3,dims=dims)
        elif opt == 4:
            batch = torch.flip(batch,dims_flip)
        elif opt == 5:
            batch = torch.rot90(torch.flip(batch,dims_flip),dims=dims)
        elif opt == 6:
            batch = torch.rot90(torch.flip(batch,dims_flip),2,dims=dims)
        elif opt == 7:
            batch = torch.rot90(torch.flip(batch,dims_flip),3,dims=dims)
    
        return batch

#calculate Minkowski metric imk with threshold thr on batch
def get_Minkowski(batch, imk, thr):
    Ns = batch.shape[0]
    bbuf = batch.cpu().numpy()
    lbuf = bbuf >= thr
    
    res = torch.zeros(Ns)
    for j in range(Ns):
        res[j] = mk.functionals(lbuf[j])[imk]
        
    return res

#calculate bin indices for given component of 2D power spectrum
#when obtaining the 1D power spectrum from the 2D power spectrum
def get_knrm_indices(knrm, kbins):
    knrm_inds = []
    knrm_Nind = []
    knrm_arrs = []
    for j in range(kbins.shape[0]+1):
        if j == 0:
            inds = np.where(knrm < kbins[j])[0]
            arr = (knrm < kbins[j]).int()
        elif j == kbins.shape[0]:
            inds = np.where(knrm > kbins[j-1])[0]
            arr = (knrm > kbins[j-1]).int()
        else:
            buf1 = np.where(knrm > kbins[j-1])[0]
            buf2 = np.where(knrm < kbins[j])[0]
            inds = np.intersect1d(buf1, buf2)
            arr = torch.div((knrm > kbins[j-1]).int() + (knrm < kbins[j]).int(),2, rounding_mode='floor')

        if j == 0:
            knrm_arrs = arr.unsqueeze(0)
        else:
            knrm_arrs = torch.cat((knrm_arrs, arr.unsqueeze(0)), dim=0)
        knrm_inds.append(inds)
        knrm_Nind.append(len(inds))
        #knrm_arrs.append(arr)
        
    knrm_Nind = torch.tensor(knrm_Nind)
    return knrm_inds, knrm_arrs, knrm_Nind   
        
