# %% paths
import os
project_root = # absolut path to the myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/kinhib')
os.chdir(working_dir)

# add higher level directories to the path
import sys
while working_dir.split('/')[-1] != 'myerson_results': working_dir = '/'.join(working_dir.split('/')[:-1])
sys.path.insert(0, working_dir)
sys.path.insert(0, os.path.join(working_dir, "training"))
sys.path.insert(0, os.path.join(working_dir, "explanations"))
working_dir = os.getcwd()

# other imports
import pickle
import visualizations as vis
import numpy as np
from clogp_ground_truth import crippen_contrib_per_atom
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import io
from PIL import Image

def rescale(myerson_dict: dict, lower=-1, upper=1) -> dict:
    min_ = min(myerson_dict.values())
    max_ = max(myerson_dict.values())

    d = {}
    for k, v in myerson_dict.items():
        x = ( (upper-lower)*(v - min_) / (max_-min_) ) + lower
        d.update({k: x})
    return d
# %% 
smi = "OC1=CC=C(C2=NC3=CC=CC=C3C4=C5C(NC=C25)=NC=C4)C(Cl)=C1"
smi_adv = "OC1=CC=C(C2=CC3=CC=CC=C3C4=C5C(CC=C25)=CC=C4)C(Cl)=C1"
with open(os.path.join(working_dir, '3KCK_no_sigm_myerson_values.pkl'), 'rb') as f:
    my_raw_3kck = pickle.load(f)
with open(os.path.join(working_dir, '3KCK_myerson_values.pkl'), 'rb') as f:
    my_3kck = pickle.load(f)
with open(os.path.join(working_dir, '3KCK_tanh_myerson_values.pkl'), 'rb') as f:
    my_tanh_3kck = pickle.load(f)

with open(os.path.join(working_dir, '3KCK_no_sigm_adversarial_myerson_values.pkl'), 'rb') as f:
    adv_my_raw_3kck = pickle.load(f)
with open(os.path.join(working_dir, '3KCK_adversarial_myerson_values.pkl'), 'rb') as f:
    adv_my_3kck = pickle.load(f)
with open(os.path.join(working_dir, '3KCK_tanh_adversarial_myerson_values.pkl'), 'rb') as f:
    adv_my_tanh_3kck = pickle.load(f)
# %%
vis.draw_with_weights(smi, list(my_3kck.values()))
vis.draw_with_weights(smi, list(rescale(my_3kck).values()))
vis.draw_with_weights(smi, list(my_tanh_3kck.values()))
vis.draw_with_weights(smi, list(my_raw_3kck.values()))
vis.draw_with_weights(smi, list(rescale(my_raw_3kck).values()))
# %%
vis.draw_with_weights(smi_adv, list(adv_my_3kck.values()))
vis.draw_with_weights(smi_adv, list(rescale(adv_my_3kck).values()))
vis.draw_with_weights(smi_adv, list(adv_my_tanh_3kck.values()))
vis.draw_with_weights(smi_adv, list(adv_my_raw_3kck.values()))
vis.draw_with_weights(smi_adv, list(rescale(adv_my_raw_3kck).values()))

# %%
vis.draw_with_weights(smi, list(my_3kck.values()), 
                      'images/a__3KCK',
                      save_format='svg')
vis.draw_colorbar(list(my_3kck.values()), 
                      'images/a__3KCK_cb',
                      save_format='svg')
# pred = 0.677102
vis.draw_with_weights(smi_adv, list(adv_my_3kck.values()),
                      'images/c__3KCK_adversarial',
                      save_format='svg')
vis.draw_colorbar( list(adv_my_3kck.values()),
                      'images/c__3KCK_adversarial_cb',
                      save_format='svg')

# pred = 0.038208