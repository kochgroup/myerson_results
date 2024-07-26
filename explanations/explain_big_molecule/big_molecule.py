# %%
import sys, os
assert os.getcwd().split('/')[-1] == "explain_big_molecule"
path_to_root = os.getcwd()
while path_to_root.split('/')[-1] != 'myerson_results': path_to_root = '/'.join(path_to_root.split('/')[:-1])
sys.path.insert(0,path_to_root)
sys.path.insert(0, os.path.join(path_to_root, "training"))

# load config
import yaml
path = # absolute path to training/projects/mw/240330-175623-add-42
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
dim_in, dim_out = dimensions.infer_dims(dataset[0], cfg['dataset']['data_type'])
model = loading.load_model(cfg['model']['name'])(dim_in, dim_out, **cfg['model'])
ckpt = torch.load(os.path.join(path, "checkpoints", "last.ckpt"))
model.load_state_dict(rename_state_dict_keys(ckpt['state_dict']))

from rdkit import Chem
from myerson import MyersonSamplingExplainer
from explanations.graph_from_smiles import GraphFromSmiles
import explanations.visualizations as vis
# %%
smi = "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@@H](NC(=O)CCCCCCCCCCCCCCCCC(O)=O)C(O)=O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)C(C)(C)NC(=O)[C@@H](N)CC1=CN=CN1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=C1C=CC=C2)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(O)=O"
mol = Chem.MolFromSmiles(smi); mol
graph = GraphFromSmiles(smi, 4113.641)

explainer = MyersonSamplingExplainer(graph, model, 42, 1000, False)
my_values = explainer.sample_all_myerson_values()

pred = explainer.calculate_prediction()

with open("mw_myerson.yaml", "w") as f:
    yaml.safe_dump({"my_values": my_values,
                    "pred": pred,
                    "gt": 4113.641})
# # %%
# pse = Chem.GetPeriodicTable()
# absolute_error = []
# for i, a in enumerate(mol.GetAtoms()):
#     wt = pse.GetAtomicWeight(a.GetSymbol())
#     my = my_values[i]
#     absolute_error.append(abs(wt - my))

# sum(absolute_error)/len(absolute_error)

# %% #######################################################
path = #absolute path to training/projects/clogp/240331-063122-gat-42
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
dim_in, dim_out = dimensions.infer_dims(dataset[0], cfg['dataset']['data_type'])
model = loading.load_model(cfg['model']['name'])(dim_in, dim_out, **cfg['model'])
ckpt = torch.load(os.path.join(path, "checkpoints", "last.ckpt"))
model.load_state_dict(rename_state_dict_keys(ckpt['state_dict']))
# %%
smi = "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@@H](NC(=O)CCCCCCCCCCCCCCCCC(O)=O)C(O)=O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)C(C)(C)NC(=O)[C@@H](N)CC1=CN=CN1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=C1C=CC=C2)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(O)=O"
mol = Chem.MolFromSmiles(smi); mol
graph = GraphFromSmiles(smi, 4113.641)

explainer = MyersonSamplingExplainer(graph, model, 42, 1000, False)
my_values = explainer.sample_all_myerson_values()

pred = explainer.calculate_prediction()

with open("clogp_myerson.yaml", "w") as f:
    yaml.safe_dump({"my_values": my_values,
                    "pred": pred,
                    "gt": 4113.641})