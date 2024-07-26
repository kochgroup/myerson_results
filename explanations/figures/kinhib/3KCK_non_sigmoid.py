assert False, "Using old version of Myerson calculator"
# %%
import sys, os
# check for correct working directory 
assert os.getcwd().split('/')[-1] == "kinhib"
# add higher level directories to the path
path_to_root = os.getcwd()
while path_to_root.split('/')[-1] != 'myerson_results': path_to_root = '/'.join(path_to_root.split('/')[:-1])
sys.path.insert(0,path_to_root)
sys.path.insert(0, os.path.join(path_to_root, "training"))

# %% load config
import yaml
path = # absolute path to training/projects/kinase_inhibitors/231010-191518-bce_logits-y1-ll_fc
with open(os.path.join(path, ".hydra", "config.yaml")) as f:
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
ckpt = torch.load(os.path.join(path, "checkpoints", "last.ckpt"))
model.load_state_dict(rename_state_dict_keys(ckpt['state_dict']))

# %%
from explanations.graph_from_smiles import GraphFromSmiles
import pickle
smi = "OC1=CC=C(C2=NC3=CC=CC=C3C4=C5C(NC=C25)=NC=C4)C(Cl)=C1"
graph = GraphFromSmiles(smi)

from explanations.MyX_v6 import MyersonCalculator

explainer = MyersonCalculator(graph, model, sampling_params={"use_sample":False},
                              verbose=True)#, visualize_mol=True)


with open("3KCK_no_sigm_myerson_values.pkl", "wb") as f:
    pickle.dump(explainer.my_values, f)
# pred = 0.677102
# %% adversarial 
smi_adversarial = "OC1=CC=C(C2=CC3=CC=CC=C3C4=C5C(CC=C25)=CC=C4)C(Cl)=C1"
graph_adversarial = GraphFromSmiles(smi_adversarial)

from explanations.MyX_v6 import MyersonCalculator

explainer = MyersonCalculator(graph_adversarial, model, sampling_params={"use_sample":False},
                              verbose=True)#, visualize_mol=True)
# pred = 0.038208

with open("3KCK_no_sigm_adversarial_myerson_values.pkl", "wb") as f:
    pickle.dump(explainer.my_values, f)