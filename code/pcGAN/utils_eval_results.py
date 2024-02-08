import numpy as np
import torch
import pickle
import scipy.linalg as splinalg
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


############## data collection
##############

#object to collect results from different runs
class Collection():
    def __init__(self, ds_opt, paths, lmodels):
        self.ds_opt = ds_opt
        self.paths = paths
        self.lmodels = lmodels
        
#model names to be displayed in the plots
def get_model_name(jm):
    if jm == 0:
        return 'WGAN'
    if jm == 1:
        return 'pcGAN'
    if jm == 3:
        return 'Wu et al.'
    if jm == 4:
        return 'WGAN-GP'
    if jm == 5:
        return 'SNGAN'



#function to combine data from different runs;
#only the arrays required for evaluation are extracted
def combine_run_data(ds_opt, paths, lmodels, par_labels=''):
    if not type(paths) is list:
        paths = [paths]
    if not type(lmodels) is list:
        lmodels = [lmodels]
        
    #initialize res
    res = Collection(ds_opt, paths, lmodels)    
    res.Nrun = 3
    res.Nmodel = 10
    res.Gges = [[] for j in range(res.Nrun)]
    res.pmetrics_hist_abs_ges_ges = [[] for j in range(res.Nrun)]
    res.constraints_hist_abs_ges_ges = [[] for j in range(res.Nrun)]
    res.mname_vec = []
    res.mname_vec2 = []
    res.model_vec = []
    
    for j, (path, models) in enumerate(zip(paths, lmodels)):
        
        print(path, models)
        fpath = path + 'files/' + 'ds' + str(ds_opt) + '_results.pk'
        with open(fpath, 'rb') as f:
            res_buf, pbuf = pickle.load(f)
            
            
            for jN in range(res_buf.pars['Nrun']):
                    
                for jm in models:
                    print(jm)
                    if jN == 0:
                        if par_labels == '':
                            res.mname_vec.append(res_buf.mname_vec[jm])
                            res.mname_vec2.append(get_model_name(jm))
                        else:
                            buf = res_buf.mname_vec[jm]
                            buf2 = ''
                            for ip, par_label in enumerate(par_labels):
                                buf += ', '
                                if ip > 0:
                                    buf2 += ', '
                                buf += par_label + '=' + str(pbuf[par_label])
                                
                                if par_label == 'bs':
                                    buf2 += par_label  + '=' + str(pbuf[par_label])
                                if par_label == 'omega_c':
                                    buf2 += r'$\lambda$='+str(pbuf[par_label])
                                if par_label == 'match_opt':
                                    buf2 += 'h='
                                    if pbuf[par_label] == 'abs':
                                        buf2 += 'TV'
                                    else:
                                        buf2 += pbuf[par_label]
                                if par_label == 'fforget':
                                    buf2 += r'$\epsilon$='+str(pbuf[par_label])
                                
                            res.mname_vec.append(buf)
                            res.mname_vec2.append(buf2)
                        res.model_vec.append(jm)
                    res.Gges[jN].append(res_buf.gnet_ges_ges[jN][jm]) #can jN be done with :?
                    res.pmetrics_hist_abs_ges_ges[jN].append(res_buf.pmetrics_hist_abs_ges_ges[jN][jm])
                    res.constraints_hist_abs_ges_ges[jN].append(res_buf.constraints_hist_abs_ges_ges[jN][jm])
                    
                    if jm == 1:
                        res.pars = pbuf
                        
    return res
#%%
def load_run_data(ds_opt, opt = 'model_comp'):
    Ns = 100000
    use_cvalue_input = False
    par_labels = ''
    if ds_opt == 1:        
        if opt == 'model_comp':
            paths = ['../results/ds1_bs256_model1/wave_forms/',
                     '../results/ds1_bs256_model3/wave_forms/',
                     '../results/ds1_bs256_model0/wave_forms/',
                     '../results/ds1_bs256_model4/wave_forms/',
                     '../results/ds1_bs256_model5/wave_forms/',
                     ]        
            lmodels = [[1], [3], [0], [4], [5]]
            
        if opt == 'bs_comp':
            paths = ['../results/ds1_KL_bs64_fforget0/wave_forms/',
                     '../results/ds1_KL_bs64_fforget0.5/wave_forms/',
                     '../results/ds1_KL_bs64_fforget0.9/wave_forms/',
                     '../results/ds1_KL_bs128_fforget0/wave_forms/',
                     '../results/ds1_KL_bs128_fforget0.5/wave_forms/',
                     '../results/ds1_KL_bs128_fforget0.9/wave_forms/',
                     '../results/ds1_KL_bs256_fforget0/wave_forms/',
                     '../results/ds1_KL_bs256_fforget0.5/wave_forms/',
                     '../results/ds1_KL_bs256_fforget0.9/wave_forms/',
                    ]
            lmodels = [[1]]*len(paths)
            par_labels = ['bs', 'fforget']
            
        if opt == 'omega_comp':
            paths = ['../results/ds1_KL_bs128_omega0.01/wave_forms/',
                     '../results/ds1_KL_bs128_omega0.1/wave_forms/',
                     '../results/ds1_KL_bs128_omega0.5/wave_forms/',
                     '../results/ds1_KL_bs128_omega1/wave_forms/',
                     '../results/ds1_KL_bs128_omega2/wave_forms/',
                     '../results/ds1_KL_bs128_omega5/wave_forms/',
                     '../results/ds1_KL_bs128_omega10/wave_forms/',
                    ]
            lmodels = [[1]]*len(paths)
            par_labels = ['omega_c']
        
        if opt == 'match_opt_comp':
            paths = ['../results/ds1_abs_bs128_fforget0.5/wave_forms/',                
                     '../results/ds1_abs_bs128_fforget0.9/wave_forms/',
                     '../results/ds1_KL_bs128_fforget0.5/wave_forms/',
                     '../results/ds1_KL_bs128_fforget0.9/wave_forms/',
                     '../results/ds1_JS_bs128_fforget0.5/wave_forms/',
                     '../results/ds1_JS_bs128_fforget0.9/wave_forms/',             
                    ]
            lmodels = [[1]]*len(paths)
            par_labels = ['match_opt', 'fforget']
            
            
            
        
        dpath = '../results/datasets/ds1.pk'
        cpath = '../results/datasets/ds1_ptrue_rep.pk'
        cinds_selection = [1, 15, 50]
    if ds_opt == 3:
        if opt == 'model_comp':
            paths = ['../results/ds3_bs256_model1/IceCube/',
                     '../results/ds3_bs256_model3/IceCube/',
                     '../results/ds3_bs256_model0/IceCube/',
                     '../results/ds3_bs256_model4/IceCube/',
                     '../results/ds3_bs256_model5/IceCube/',
                     ]        
            lmodels = [[1], [3], [0],[4], [5]]
        
        
        dpath = '../results/datasets/ds3.pk'
        cpath = '../results/datasets/ds3_ptrue_rep.pk'
        cinds_selection = [0,1]
        

    fname = '../results/datasets/ds' + str(ds_opt) + '.pk'
    with open(fname, 'rb') as f:
        ds = pickle.load(f)
    
    fname = '../results/datasets/ds' + str(ds_opt) + '_ptrue_KDE.pk'
    with open(fname, 'rb') as f:
        cebm = pickle.load(f)
        
    res = combine_run_data(ds_opt, paths, lmodels, par_labels = par_labels)
    res.cinds_selection = cinds_selection
    
    return ds, cebm, res 









##############
##############
#FID scores
def calculate_FID(mu, Sigma, gmu, gSigma):
    return torch.abs(mu-gmu)**2 + torch.trace(Sigma + gSigma - 2*splinalg.sqrtm(Sigma@gSigma))
    
def get_FID(data, gdata):
    mu = data.mean(0).cpu()
    Sigma = torch.cov(data.squeeze().T).cpu()

    gmu = gdata.mean(0).cpu()
    gSigma = torch.cov(gdata.squeeze().T).cpu()        
    FID = calculate_FID(mu, Sigma, gmu, gSigma).cpu().numpy().sum()

    return FID




##############
##############
# precision and recall
    
# determine distance to k-th nearest neighbor
def get_kdist(dsamples, k=1):
    Ns = dsamples.shape[0]
    kdist = np.zeros(Ns)
    
    for j in range(Ns):
        kdist[j] = np.partition(np.sum((dsamples[j] - dsamples)**2,1),k)[k] #partition is faster than sort!
    return kdist


# calculate precision
def get_precision(gsamples, dsamples, kdist):
    Ngs = gsamples.shape[0]
    indist = np.zeros(Ngs)
    for j in range(Ngs):
        dists = np.sum((gsamples[j] - dsamples)**2,1) - kdist
        Nwithin = np.sum(dists<=0)
        indist[j] = 1 if Nwithin > 0 else 0
        
    return np.mean(indist), indist


#calculate recall
def get_recall(gsamples, dsamples, k=1):
    kdist = get_kdist(gsamples, k)
    recall, indist = get_precision(dsamples, gsamples, kdist)
    return recall, indist


#binary classification via kNN
def get_binary_kNN(dsamples, gsamples, k = 1):
    Nd = dsamples.shape[0]
    Ng = gsamples.shape[0]
    jd = int(Nd*0.8)
    jg = int(Ng*0.8)
    
    #collect data
    x_train = np.concatenate((dsamples[:jd], gsamples[:jg]),axis=0)
    y_train = np.concatenate((np.ones(jd), np.zeros(jg)), axis=0)

    x_test = np.concatenate((dsamples[jd:], gsamples[jg:]),axis=0)
    y_test = np.concatenate((np.ones(Nd-jd), np.zeros(Ng-jg)), axis=0)
    
    
    #from sklearn.neighbors import KNeighborsClassifier
    ckNN = KNeighborsClassifier(n_neighbors=k)
    ckNN.fit(x_train, y_train)
    yhat_test = ckNN.predict(x_test)

    return np.mean(y_test == yhat_test)





#############
#############

#plot box_plots/scatter plots of run results
def plot_box_plots(eval_metrics, eval_names, model_names, pname='', popt='scatter'):
    Nmodels = len(model_names)
    xticks = [j for j in range(1,Nmodels+1)]
    
    Neval =  len(eval_metrics)
    Nrow = (Neval-1)//4+1
    Ncolumn = min(4,Neval)
    wcolumn = 5
    fig, axs = plt.subplots(Nrow, Ncolumn, figsize = (Ncolumn*wcolumn, Nrow*3.5))
    for j in range(Neval):
        ax = axs[j//Ncolumn, j%Ncolumn] if Nrow > 1 else axs[j%Ncolumn]
        if popt == 'box':
            ax.boxplot(eval_metrics[j])
        if popt == 'scatter':
            x = np.array(xticks*eval_metrics[j].shape[0])
            ax.scatter(x, eval_metrics[j].reshape(-1), color='k', marker='x')
        ax.set_xticks(xticks, model_names, rotation=90)
        ax.set_title(eval_names[j])
        
    fig.subplots_adjust(hspace=0.5)
    if pname!='':
        plt.savefig(pname+'.pdf', bbox_inches='tight')
    plt.show()