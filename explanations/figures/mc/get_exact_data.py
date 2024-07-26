# use kinase dataset; 
# sort dataset by size;
# calculate 10 myerson/exact for every size
# vary stepsize also 

# %% paths
import os
project_root = # absolute path to myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/mc')
os.chdir(working_dir)

# add higher level directories to the path
import sys
while working_dir.split('/')[-1] != 'myerson_results': working_dir = '/'.join(working_dir.split('/')[:-1])
sys.path.insert(0, working_dir)
sys.path.insert(0, os.path.join(working_dir, "training"))
sys.path.insert(0, os.path.join(working_dir, "explanations"))
working_dir = os.getcwd()

# other imports
import time
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
from myerson import MyersonExplainer, MyersonSamplingExplainer
# %% load config
import yaml
path = # path to training/projects/kinase_inhibitors/231010-191518-bce_logits-y1-ll_fc
with open(os.path.join(path, ".hydra", "config.yaml")) as f:
    cfg = yaml.safe_load(f)
# load dataset
from training.utils import loading
from training.utils import shuffle
from training.utils import splitting
dataset_path = os.path.join(cfg['dataset']['dir'], cfg['dataset']['name'])
dataset = loading.load_dataset(cfg['dataset']['name'])(root=dataset_path, **cfg['dataset'])
dataset = shuffle.shuffle_dataset(dataset, cfg['dataset']['shuffle_seed'])
trainset, valset, testset = splitting.random_split(dataset, cfg['dataset']['split'])

# load model
from training.utils import dimensions
from explanations.rename_state_dict_keys import rename_state_dict_keys
import torch
device = 'cuda'
dim_in, dim_out = dimensions.infer_dims(dataset[0], cfg['dataset']['data_type'])
model = loading.load_model(cfg['model']['name'])(dim_in, dim_out, **cfg['model'])
ckpt = torch.load(os.path.join(path, "checkpoints", "last.ckpt"))
model.load_state_dict(rename_state_dict_keys(ckpt['state_dict']))
class SigmoidModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = self.model(x, edge_index, batch, *args, **kwargs)
        return torch.nn.functional.sigmoid(x)
model = SigmoidModel(model).to(device)

# %%
size_idxs = {}
for i, d in tqdm(enumerate(dataset), total=len(dataset)):
    size = d.x.shape[0]
    if size not in size_idxs.keys():
        size_idxs.update({size: [i]})
    else:
        size_idxs[size].append(i)

# %%
# len(sorted(list(size_idxs.keys())))
keys_less_10 = []
for k,v in size_idxs.items():
    if len(v) < 10: keys_less_10.append(k)
print(keys_less_10)
for k in keys_less_10:
    del size_idxs[k]

# %% exact: do calc up to 26
my_exact = {}
for i in range(9, 27):
    print(f"Calculate for molecules size {i}:")
    helper = {}
    for idx in tqdm(size_idxs[i][:10]):
        start = time.time()
        explainer = MyersonExplainer(dataset[idx], model)
        my = explainer.calculate_all_myerson_values()
        t = time.time() - start
        d = {'my_values': my, 'smiles': dataset[idx].smiles,
             'pred': explainer.calculate_prediction(), 'time': t}
        helper.update({idx: d})
    my_exact.update({i: helper})
    with open('data/exact_my.yaml', 'a') as f:
        yaml.safe_dump({i:helper}, f)
