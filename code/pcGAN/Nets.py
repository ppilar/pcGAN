# -*- coding: utf-8 -*-
import torch
import torch.nn as nn



#######################
#######################
#######################

#fully connected NN
class Net(nn.Module):
    def __init__(self, Uvec = [40]*5, iovec = [1,1], Npar=3, fdrop = 0, bounds = (0,0,0,0), act = 'tanh', normalize = True, device = 'cpu'):
        super(Net, self).__init__()    
        
        self.Uvec = Uvec
        #self.act = act
        self.normalize = normalize
        self.device = device
        
        #if type(lb) is tuple:    
        self.lb = torch.tensor(bounds[0]).float().to(device)
        self.ub = torch.tensor(bounds[1]).float().to(device)
        
        self.ylb = torch.tensor(bounds[2]).float().to(device)
        self.yub = torch.tensor(bounds[3]).float().to(device)
        
        current_dim = iovec[0]
        self.layers = nn.ModuleList()
        for j, hdim in enumerate(self.Uvec):
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, iovec[1]))
        
        self.dr = nn.Dropout(fdrop)
        
        if act == 'tanh':  self.act = torch.tanh
        elif act == 'relu': self.act = torch.relu
        
        
     
    def forward(self, X): 
        if type(X) is tuple:
            X = X[0]
        
        if self.normalize == True:
            X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input   
        
        
        for j, layer in enumerate(self.layers[:-1]):
            X = self.act(layer(X))
            
        X = self.dr(X)  
        X = self.layers[-1](X)
        
        if self.normalize == True:
            X = 0.5*((X + 1.0)*(self.yub - self.ylb)) + self.ylb

        return X

#conditional fully connected NN
class cNet(nn.Module): #map from t to x
    def __init__(self, Uvec = [40]*5, iovec = [1,1], Npar=3, fdrop = 0, bounds = (0,0,0,0), act = 'tanh', normalize = True, device = 'cpu'):
        super(cNet, self).__init__()
 
        self.device = device
        self.Nc = 101
        
        self.embed = nn.Embedding(self.Nc, 30)
        self.fce = nn.Linear(30,15)
    
        self.Uvec = Uvec
        self.act = act
        self.normalize = normalize
        self.device = device
        
        self.bounds = bounds
        
        self.ylb = torch.tensor(self.bounds[2]).float().to(device)
        self.yub = torch.tensor(self.bounds[3]).float().to(device)
        
        current_dim = iovec[0]
        self.layers = nn.ModuleList()
        for j, hdim in enumerate(self.Uvec):
            if j == 1:
                current_dim += 15
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, iovec[1]))
        
        self.dr = nn.Dropout(fdrop)
        
     
    def forward(self, X):
        cl = X[1].long()
        X = X[0]
        
        #get bounds        
        self.lb = self.bounds[0][cl.long()].float().to(X.device).unsqueeze(1)
        self.ub = self.bounds[1][cl.long()].float().to(X.device).unsqueeze(1)
        
        
        #embed
        xcl = self.embed(cl)
        xcl = torch.relu(self.fce(xcl))
        
        #normalize        
        if type(X) is tuple:
            X = X[0]
        
        if self.normalize == True:
            X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input   
        
        if self.act == 'tanh':  act = torch.tanh
        elif self.act == 'relu': act = torch.relu
        
        for j, layer in enumerate(self.layers[:-1]):
            X = act(layer(X))
            if j == 0:
                X = torch.cat((X, xcl), 1)
            
        X = self.dr(X)  
        X = self.layers[-1](X)
        
        if self.normalize == True:
            X = 0.5*((X + 1.0)*(self.yub[cl].unsqueeze(1) - self.ylb)) + self.ylb

        return X
    


#######################
#######################
#######################

#generator network for waveform dataset
class Generator_1D_large(nn.Module):
    def __init__(self, ds=0, xvec=-1, nopt = 'small', latent_dim = 20, device='cpu'):
        super(Generator_1D_large, self).__init__()
        
        self.xvec = xvec
        self.latent_dim = latent_dim
        self.NB = 4
        self.GBlocks = nn.ModuleList()
        
        stride = 2
        ct1 = (25 - (latent_dim - 1)*stride)
        # if ct1 < 0:
        #     opp = -ct1
        #     ct1 = 1
            
        #25 = (latent_dim - 1)*stride + 1*(ct1-1) + op +1
        if nopt == 'small':
            self.GBlocks.append(GBlock([1,80], ct_stride=ct1, Nlc=0, output_padding=opp, normalize=True))
            self.GBlocks.append(GBlock([80,50], ct_stride=2, Nlc=0, normalize=True))
            self.GBlocks.append(GBlock([50,25], ct_stride=2, Nlc=0, output_padding=0, normalize=True))
            self.GBlocks.append(GBlock([25,10,1], ct_stride=2, Nlc=0, normalize=True))
            self.cf = nn.Conv1d(10, 1, 3, padding='same', stride=1)
        if nopt == 'big':
            Nlc = 2
            self.GBlocks.append(GBlock([1,256], ct_stride=ct1, Nlc=Nlc, normalize=True, latent_dim = latent_dim))
            self.GBlocks.append(GBlock([256,128], ct_stride=2, Nlc=Nlc, normalize=True, latent_dim = latent_dim))
            self.GBlocks.append(GBlock([128,64], ct_stride=2, Nlc=Nlc, output_padding=0, normalize=True, latent_dim = latent_dim))
            self.GBlocks.append(GBlock([64,32], ct_stride=2, Nlc=Nlc, normalize=True, latent_dim = latent_dim))
            self.cf = nn.Conv1d(32, 1, 3, padding='same', stride=1)
     
    def forward(self, x):
        #print('Generator')
        if x.ndim == 2:
            x = x.unsqueeze(1)
        #print(x.shape)
        for j in range(self.NB):
            x = self.GBlocks[j](x)

        x = self.cf(x)
        x = torch.tanh(x)
        return x.squeeze(1)
    
#discriminator network for waveform dataset
class Discriminator_1D_large(nn.Module):
    def __init__(self, ds=0, xvec=-1, nopt = 'small', device='cpu', use_fft_input = False):
        super(Discriminator_1D_large, self).__init__()
        
        self.xvec = xvec
        
        Nci = 1 if use_fft_input == False else 2
        self.NB = 4
        self.DBlocks = nn.ModuleList()
        if nopt == 'small':            
            self.DBlocks.append(DBlock([Nci,16], 4, stride=2, Nlc=0, normalize=True))
            self.DBlocks.append(DBlock([16,32], 4, stride=2, Nlc=0, normalize=True))
            self.DBlocks.append(DBlock([32,64], 4, stride=2, Nlc=0, normalize=False))
            self.DBlocks.append(DBlock([64,128], 4, stride=2, Nlc=0, normalize=False))
            self.fc = nn.Linear(128*10, 1)
        if nopt == 'big':
            Nlc = 2 
            self.DBlocks.append(DBlock([Nci,32],4, 2, Nlc=Nlc, normalize=True))
            self.DBlocks.append(DBlock([32,64],4, 2, Nlc=Nlc, normalize=True))
            self.DBlocks.append(DBlock([64,128],4, 2, Nlc=Nlc, normalize=False))
            self.DBlocks.append(DBlock([128,256],4, 2, Nlc=Nlc, normalize=False))   
            self.fc = nn.Linear(256*10,1)
            
        
    def forward(self, x):
        #print('Discriminator')
        if type(x[1]) != int:
            xfft = x[1]
            xfft2 = torch.cat((torch.real(xfft)[:,1:], torch.imag(xfft)[:,1:]),1)
            x = torch.cat((x[0].unsqueeze(1),xfft2.unsqueeze(1)),1)
        else:        
            x = x[0]
            
        if x.ndim == 2:
            x = x.unsqueeze(1)        
        
        for j in range(self.NB):
            x = torch.relu(self.DBlocks[j](x))

        
        x = nn.Flatten()(x)
        x = self.fc(x)
        
        return x
    
    
#block used in generator network
class GBlock(nn.Module):
    def __init__(self, Nc_io, ct_stride=2, Nlc = 0, output_padding=0, normalize=True, latent_dim = -1):        
        super(GBlock, self).__init__()
        #Nc_io ... number of in/out channels
        #stride ... stride of ConvTranspose1d
        #Nlc ... number of convolutional layers
        
        self.act = torch.tanh
        self.fnorm = torch.nn.BatchNorm1d     
        self.normalize = normalize
        
        self.Nlc = Nlc
        
        #ct1 = (25 - (latent_dim - 1)*stride)
        stride = 2
        if ct_stride < 0:
            stride = 1
            ct_stride = (25 - (latent_dim - 1)*stride)
            #opp = -ct1
            #ct1 = 1
        self.ct = nn.ConvTranspose1d(Nc_io[0], Nc_io[1], ct_stride, output_padding=output_padding, stride=stride)
        self.bn0 = self.fnorm(Nc_io[0])
        
        while(len(Nc_io) <= Nlc+1):
            Nc_io.append(Nc_io[-1])
        
        self.lconv = nn.ModuleList()
        self.bns = nn.ModuleList()
        for j in range(Nlc):
            self.lconv.append(nn.Conv1d(Nc_io[j+1], Nc_io[j+2], 3, padding='same', stride=1))
            self.bns.append(self.fnorm(Nc_io[j+1]))
        
        
        
    def forward(self, x):
        if self.normalize == True:
            x = self.bn0(x)
        x = self.act(self.ct(x))
        
        for j in range(self.Nlc):
            if self.normalize == True:
                x = self.bns[j](x)
            x = self.act(self.lconv[j](x))    
        return x

#block used in discriminator network
class DBlock(nn.Module):
    def __init__(self, Nc_io, kernel_size, stride, Nlc = 0, normalize = True):        
        super(DBlock, self).__init__()
        #Nc_io ... number of in/out channels
        #kernel_size ... size of conv kernel
        #stride ... stride of first convolution in bock; other convs preserve size
        #Nlc ... number of size-preserving conv layers
        #normalize ... whether layers within a block should be normalized
        
        
        self.act = torch.relu
        #self.act = torch.nn.LeakyReLU(0.2)
        self.fnorm = torch.nn.InstanceNorm1d        
        self.normalize = normalize
        
        self.Nlc = Nlc
        self.conv0 = nn.Conv1d(Nc_io[0], Nc_io[1], kernel_size, stride=stride)
        self.bn0 = self.fnorm(Nc_io[0])
        
        while(len(Nc_io) <= Nlc+1):
            Nc_io.append(Nc_io[-1])
        
        self.lconv = nn.ModuleList()
        self.bns = nn.ModuleList()
        for j in range(Nlc):
            self.lconv.append(nn.Conv1d(Nc_io[j+1], Nc_io[j+2], kernel_size, padding='same', stride=1))
            self.bns.append(self.fnorm(Nc_io[j+1]))
            
    def forward(self, x):
        if self.normalize:
            x = self.bn0(x)
        x = self.act(self.conv0(x))  
        
        for j in range(self.Nlc-1):
            if self.normalize:
                x = self.bns[j](x)
            x = self.act(self.lconv[j](x))            
            #print(x.shape)
        
        if self.Nlc > 0:
            x = self.lconv[-1](x)  
            
        return x
        
        
        
#############
############# CAMELS GAN architecture

# ConvTranspose2d(channels_in, channels_out, kernel, stride, padding)
class Generator_64(nn.Module):
    def __init__(self, Z_DIM, G_HIDDEN):
        super(Generator_64, self).__init__()
        self.main = nn.Sequential(
            # 1st layer (input: 100x1x1 ----> output: 512x4x4)
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.LeakyReLU(0.2,inplace = True),
            # 2nd layer (input: 512x4x4 ----> output: 256x8x8)
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.LeakyReLU(0.2,inplace = True),
            # 3rd layer (input: 256x8x8 ----> output: 128x16x16)
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.LeakyReLU(0.2,inplace = True),
            # 4th layer (input: 128x16x16 ----> output: 64x32x32)
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.LeakyReLU(0.2,inplace = True),
            # output layer (input: 64x32x32 ----> 1x64x64)
            nn.ConvTranspose2d(G_HIDDEN, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, gin):
        gin = gin.unsqueeze(2).unsqueeze(3) 
        return self.main(gin).squeeze(1)


#Conv2d(channels_in, channels_out, kernel, stride, padding)
class Discriminator_64(nn.Module):
    def __init__(self, D_HIDDEN):
        super(Discriminator_64, self).__init__()
        self.main = nn.Sequential(
            # 1st layer (input: 1x64x64 ----> output: 64x32x32)
            nn.Conv2d(1, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer (input: 64x32x32 ----> output: 128x16x16)
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer (input: 128x16x16 ----> output: 256x8x8)
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer (input: 256x8x8 ----> output: 512x4x4)
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer (input: 512x4x4 ----> output: 1x1x1)
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid() #Returns a number from 0 to 1: probability
        )

    def forward(self, din):
        din = din[0].unsqueeze(1)
        return self.main(din).view(-1, 1).squeeze(1)


#######################
#######################
######## Ice Cube
class Discriminator_IceCube_v0(nn.Module):
    #(almost) architecture of master's thesis of aholmberg
    def __init__(self):
        super(Discriminator_IceCube_v0, self).__init__()
        
        af0 = [5,15,25,35]
        nfilter = 32
        self.aconv0 = nn.ModuleList()
        for j in range(len(af0)):
            self.aconv0.append(nn.Conv1d(1, nfilter, af0[j], stride=4))
            
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(nfilter, 1, 1)
        
        afc = [182, 92, 45, 20, 1]
        self.afc = nn.ModuleList()
        for j in range(len(afc)-1):
            self.afc.append(nn.Linear(afc[j],afc[j+1]))
        
    def forward(self, x):
        x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        ax0 = []
        for j in range(len(self.aconv0)):
            buf = self.act(self.aconv0[j](x))
            ax0.append(buf)
        
        x1 = torch.cat(ax0,dim=2) 
        x1 = self.act(self.conv1(x1))
        
        
        for j in range(len(self.afc)-1):
            x1 = self.act(self.afc[j](x1))
        x1 = self.afc[-1](x1)
        
        return x1
        
        
class Generator_IceCube_v0(nn.Module):
    #similar to architecture of master's thesis of aholmberg
    def __init__(self, latent_dim = 20):
        super(Generator_IceCube_v0, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc0 = nn.Linear(latent_dim, 24)
        
        
        af = [48,24,12,6]
        op = [0,0,0]
        self.conv0 = nn.Conv1d(1,af[0], kernel_size=3, stride = 1, padding='same')
        self.aconv = nn.ModuleList()
        self.aconvT = nn.ModuleList()
        for j in range(len(af)-1):
            self.aconv.append(nn.Conv1d(af[j],af[j], kernel_size=3, stride=1, padding='same'))
            self.aconvT.append(nn.ConvTranspose1d(af[j],af[j+1], kernel_size=3, stride=2, output_padding=op[j]))
            
        self.fconv = nn.Conv1d(af[-1],1,4,1, padding=2, padding_mode='reflect')# padding='same')
        
        self.act = nn.ReLU()
        
        
    def forward(self, x):
        
        x = self.fc0(x).unsqueeze(1)
        x = self.act(self.conv0(x))
        for j in range(len(self.aconv)):
            x = self.act(self.aconv[j](x))
            x = self.act(self.aconvT[j](x))
        
        x = self.fconv(x)
        
        return x.squeeze()
    
    
    
    
    
    
        
        