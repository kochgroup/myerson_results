# %%
import os
project_root = # absolute path to myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/adversarial')
os.chdir(working_dir)

# add higher level directories to the path
import sys
while working_dir.split('/')[-1] != 'myerson_results': working_dir = '/'.join(working_dir.split('/')[:-1])
sys.path.insert(0, working_dir)
sys.path.insert(0, os.path.join(working_dir, "training"))
sys.path.insert(0, os.path.join(working_dir, "explanations"))
working_dir = os.getcwd()
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import visualizations as vis
from explanations.graph_from_smiles import GraphFromSmiles

# %%
import yaml
path = # absolute path to training/projects/kinase_inhibitors/231010-191518-bce_logits-y1-ll_fc'
with open(os.path.join(config_path, ".hydra", "config.yaml")) as f:
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
ckpt = torch.load(os.path.join(config_path, "checkpoints", "last.ckpt"))
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
with open('mc_my_data.pkl', 'rb') as f:
    data = pickle.load(f)

# %% get predictions
preds = []
for graph in tqdm([GraphFromSmiles(s) for s in data['adv_smiles']]):
    graph.to(device)
    out = model(graph.x, graph.edge_index, torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device))
    preds.append(out.item())
og_preds = []
for graph in tqdm([GraphFromSmiles(s) for s in data['original_smiles']]):
    graph.to(device)
    out = model(graph.x, graph.edge_index, torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device))
    og_preds.append(out.item())
# %%
positive_idxs = [i for i, x in enumerate(preds) if x > 0.5]
positive_preds = [x for x in preds if x > 0.5]
for i, x in enumerate(preds):
    if x < 0.5: print(i, x)
sum(positive_preds)/len(positive_preds)
# %%
idx = 35
# 95: 338, 51: 377 48.99: 416 49.87: 35
key = data['tp_keys'][idx]
og_smi = data['original_smiles'][idx]
adv_smi = data['adv_smiles'][idx]
my_og = data['my_original'][idx]
my_adv = data['my_adv'][idx]

vis.draw_with_weights(og_smi, list(my_og.values()))
vis.draw_with_weights(adv_smi, list(my_adv.values()))