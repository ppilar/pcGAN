# -*- coding: utf-8 -*-

import time
import torch
import numpy as np

#class to store results
class Results():
    def __init__(self, ds_opt, eval_it = 500, plot_it = 2500, schedule_it = 100000, fsched = 0.2, Nmodels = 4, Neval = 20000):
        self.ds_opt = ds_opt
        self.Nmodels = Nmodels
        self.Neval = Neval
        
        self.losses = [[] for j in range(Nmodels)]
        self.constraints_ges0 = [[] for j in range(Nmodels)]
        self.constraints_ges = [[] for j in range(Nmodels)]
        self.pmetrics_ges = [[] for j in range(Nmodels)]
        self.gnet_ges =  [[] for j in range(Nmodels)]
        self.dnet_ges =  [[] for j in range(Nmodels)]
        
        self.eval_it = eval_it
        self.plot_it = plot_it
        self.schedule_it = schedule_it
        self.fsched = fsched
        