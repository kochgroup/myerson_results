# %% paths
import os
project_root = # absolute path to myerson_results repository

working_dir = os.path.join(project_root, 'explanations/figures/kinhib')
data_dir = os.path.join(project_root, 'explanations/kinase_inhibitors')
os.chdir(working_dir)

# add higher level directories to the path
import sys
while working_dir.split('/')[-1] != 'mjyerson_results': working_dir = '/'.join(working_dir.split('/')[:-1])
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
import yaml
from rdkit import Chem

def get_num_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumAtoms()

# %%
files = [x for x in os.listdir(data_dir) if '.yaml' in x]
my_files = [x for x in files if x.startswith('explanations_val_')]
mymc_files = [x for x in files if x.startswith('mc_explanations_val_')]
mymc_files.remove('mc_explanations_val_upto_25.yaml')
my_files.sort()
mymc_files.sort()

my_values = {}
for file in tqdm(my_files):
    with open(os.path.join(data_dir, file), 'r') as f:
        data = yaml.safe_load(f)
    for k, v in data.items():
        my_values.update({k: v})
    
mymc_values = {}
for file in tqdm(mymc_files):
    with open(os.path.join(data_dir, file), 'r') as f:
        data = yaml.safe_load(f)
    for k, v in data.items():
        mymc_values.update({k: v})

assert len(my_values) == len(mymc_values)
for k, k2 in zip(my_values.keys(), mymc_values.keys()):
    assert k == k2
# %%
conf_mat = {'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,}
for k in tqdm(my_values.keys()):
    # same for mymc_values (I tested it)
    entry = my_values[k]['confusion_matrix_entry']
    conf_mat[entry] += 1
sum(conf_mat.values())
# %%
correlation = []
for k in tqdm(my_values.keys()):
    my = list(my_values[k]['My_sigmoid'].values())
    my_mc = list(mymc_values[k]['My_sigmoid'].values())
    correlation.append(pearsonr(my, my_mc)[0])

# for i in range(len(my)): print(my[i], my_mc[i])
sum(correlation)/len(correlation)
