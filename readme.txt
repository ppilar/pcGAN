### Prerequisites

The python environment required to run the code is given in the pcGAN.yml file.


### Experiments

In the folder code:

... to generate the results:
run ds1_main.py to generate results for Figs. 2,3
run ds1_experiments.py to generate results for Figs. 4,5


... to plot the figures (after generating the results):
Fig. 2: run plot_results.py; set ds_opt = 1, opt_list = ['model_comp']
Fig. 3: run evaluate_results.py; set ds_opt = 1, opt_list = ['model_comp']
Fig. 4: run evaluate_results.py; set ds_opt = 1 and opt_list = ['bs_comp', 'omega_comp']
Fig. 5: run evaluate_results.py; set ds_opt = 1 and opt_list = ['match_opt_comp']
