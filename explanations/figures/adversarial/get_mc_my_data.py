# %%
import sys, os
# check for correct working directory 
assert os.getcwd().split('/')[-1] == "adversarial"
import yaml
import pandas as pd
# add higher level directories to the path
path_to_root = os.getcwd()
while path_to_root.split('/')[-1] != 'myerson_results': path_to_root = '/'.join(path_to_root.split('/')[:-1])
sys.path.insert(0,path_to_root)
sys.path.insert(0, os.path.join(path_to_root, "training"))
import explanations.visualizations as vis
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import pandas as pd
from myerson import MyersonSamplingExplainer
# %%
import yaml
config_path = # absolute path to training/projects/kinase_inhibitors/231010-191518-bce_logits-y1-ll_fc
with open(os.path.join(config_path, ".hydra", "config.yaml")) as f:
    cfg = yaml.safe_load(f)
# %% load dataset
from training.utils import loading
from training.utils import shuffle
from training.utils import splitting
dataset_path = os.path.join(cfg['dataset']['dir'], cfg['dataset']['name'])
dataset = loading.load_dataset(cfg['dataset']['name'])(root=dataset_path, **cfg['dataset'])
dataset = shuffle.shuffle_dataset(dataset, cfg['dataset']['shuffle_seed'])
trainset, valset, testset = splitting.random_split(dataset, cfg['dataset']['split'])

# %% load model
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
from explanations.graph_from_smiles import GraphFromSmiles
df = pd.read_csv('tp_adv_smiles_valset.txt')

sampled_my_original = []
sampled_my_adversarial = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    og_smi = row['original_smiles']
    graph = GraphFromSmiles(og_smi)
    explainer = MyersonSamplingExplainer(graph, model)
    my_og = explainer.sample_all_myerson_values()
    sampled_my_original.append(my_og)
    adv_smi = row['adv_smiles']
    graph = GraphFromSmiles(adv_smi)
    explainer = MyersonSamplingExplainer(graph, model)
    my_adv = explainer.sample_all_myerson_values()
    sampled_my_adversarial.append(my_adv)

import pickle
data = {'tp_keys': df['tp_keys'],
        'original_smiles': df['original_smiles'],
        'adv_smiles': df['adv_smiles'],
        'my_original': sampled_my_original,
        'my_adv': sampled_my_adversarial,
        }
with open('mc_my_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# %%