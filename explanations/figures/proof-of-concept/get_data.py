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
import yaml
from training.utils import loading
from training.utils import shuffle
from training.utils import splitting
from training.utils import dimensions
from explanations.rename_state_dict_keys import rename_state_dict_keys
import torch
from myerson import MyersonExplainer
from tqdm import tqdm, trange
import pickle

def load_all(run_name:str):
    # load run config
    training_run_dir = os.path.join(project_root, 'training/projects/', run_name)
    with open(os.path.join(training_run_dir, ".hydra", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    # load run dataset
    dataset_path = os.path.join(cfg['dataset']['dir'], cfg['dataset']['name'])
    dataset = loading.load_dataset(cfg['dataset']['name'])(root=dataset_path, **cfg['dataset'])
    dataset = shuffle.shuffle_dataset(dataset, cfg['dataset']['shuffle_seed'])
    trainset, valset, testset = splitting.random_split(dataset, cfg['dataset']['split'])

    # load run model
    dim_in, dim_out = dimensions.infer_dims(dataset[0], cfg['dataset']['data_type'])
    model = loading.load_model(cfg['model']['name'])(dim_in, dim_out, **cfg['model'])
    ckpt = torch.load(os.path.join(training_run_dir, "checkpoints", "last.ckpt"))
    model.load_state_dict(rename_state_dict_keys(ckpt['state_dict']))
    return model, (trainset, valset, testset)

# %% get data for run
runs = [
'clogp/240331-052347-gcn-43',
'clogp/240331-070735-gat-43',
]
for r in runs:
    model, (trainset, valset, testset) = load_all(r)
    set_ = valset


    myerson = {}
    for i, graph in tqdm(enumerate(set_), total=len(set_)):
        explainer = MyersonExplainer(graph, model)
        my_values = explainer.calculate_all_myerson_values()
        pred = explainer.calculate_prediction()

        graph_results = {}
        graph_results.update({'smiles': graph.smiles})
        graph_results.update({'MyX': my_values})
        graph_results.update({'label': graph.y.item()})
        graph_results.update({'pred': pred})
        myerson.update({i: graph_results})

    run_name = r.split('/')[1]
    with open(f'data/{run_name}_clogp_valset.pkl', "wb") as f:
        pickle.dump(myerson, f)
