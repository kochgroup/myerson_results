# %% paths
import os
project_root = # absolute path to myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/proof-of-concept')
os.chdir(working_dir)

# add higher level directories to the path
import sys
while working_dir.split('/')[-1] != 'myerson_results': working_dir = '/'.join(working_dir.split('/')[:-1])
sys.path.insert(0, working_dir)
sys.path.insert(0, os.path.join(working_dir, "training"))
sys.path.insert(0, os.path.join(working_dir, "explanations"))

# other imports
import pickle
import visualizations as vis
import numpy as np
from clogp_ground_truth import literal_weight_per_atom
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import io
from PIL import Image

# %% load_data
with open(f'data/mw/240330-175623-add-42_mw_valset.pkl', "rb") as f:
    add_val = pickle.load(f)
with open(f'data/mw/240330-194312-mean-42_mw_valset.pkl', "rb") as f:
    mean_val = pickle.load(f)
with open(f'data/mw/240330-211538-max-42_mw_valset.pkl', "rb") as f:
    max_val = pickle.load(f)

def best_predictions(data):
    errors = np.zeros(len(data))
    for k,v in data.items():
        errors[k] = abs(data[k]['label'] - data[k]['pred'])
    return np.argsort(errors)

best_add_val = best_predictions(add_val)
best_mean_val = best_predictions(mean_val)
best_max_val = best_predictions(max_val)

# %%
def get_best_figures(ordered_dataset, dataset, how_many=10):
    for i, x in enumerate(ordered_dataset[:how_many]):
        smiles = dataset[x]['smiles']
        weights = list(dataset[x]['MyX'].values())
        try: 
            gt = literal_weight_per_atom(smiles)
        except:
            print(f"failed for {x}")
            continue
        vis.draw_with_weights(smiles, weights, 
                            f"images/def__{x}",
                            save_format='svg',
                            cmap=vis.green_purple)
        print(f"{i}: {x}")
        print(f"label: {dataset[x]['label']}")
        print(f"pred: {dataset[x]['pred']}")
        vis.draw_colorbar(weights,
                          f"images/def__{x}_cb",
                          save_format='svg',
                          cmap=vis.green_purple)
        vis.draw_with_weights(smiles, gt, 
                            f"images/def__{x}gt",
                            save_format='svg',
                            cmap=vis.yellow_blue)
        vis.draw_colorbar(gt,
                        f"images/def__{x}gt_cb",
                        cmap=vis.yellow_blue,
                        save_format='svg')
# %%                        
get_best_figures(best_mean_val, mean_val, 5)
# chosen for figure: 3132

# get_best_figures(best_add_val, add_val, 5)
# get_best_figures(best_max_val, max_val, 5)

# %%
def get_corresponding_fig(idx, dataset):
    smiles = dataset[idx]['smiles']
    weights = list(dataset[idx]['MyX'].values())
    print(f"label: {dataset[idx]['label']}")
    print(f"pred: {dataset[idx]['pred']}")
    vis.draw_with_weights(smiles, weights, 
                        f"images/def__{idx}add",
                        save_format='svg',
                        cmap=vis.green_purple)
    vis.draw_colorbar(weights,
                        f"images/def__{idx}add_cb",
                        save_format='svg',
                        cmap=vis.green_purple)
get_corresponding_fig(3132, add_val)

# %%
example = add_val[3132]
my_values = list(example['MyX'].values())
smiles = example['smiles']
gt = literal_weight_per_atom(smiles)
error = [abs(x - y) for x, y in zip(my_values, gt)]
vis.draw_bw_with_values(smiles, error, "images/g__3132")
error
my_values

# %% print label and pre data
from rdkit import Chem
from rdkit.Chem import Descriptors
idx = 3132
print(f"3132 mean\npred: {mean_val[idx]['pred']}\nlabel: {mean_val[idx]['label']}")
print(f"3132 add\npred: {add_val[idx]['pred']}\nlabel: {add_val[idx]['label']}")
mol_wt = Descriptors.MolWt(Chem.MolFromSmiles(mean_val[idx]['smiles']))
print(f"ground truth {mol_wt}")

# %% get mean errors 
# add_val, mean_val, max_val

def to_np(idx, dataset):
    return np.array(list(dataset[idx]['MyX'].values()))
def weight_np(idx, dataset):
    return np.array(literal_weight_per_atom(dataset[idx]['smiles']))


to_np(0, add_val) - weight_np(0, add_val)