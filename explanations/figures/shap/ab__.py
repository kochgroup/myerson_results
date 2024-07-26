# %% paths
import os
project_root = # absolute path to myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/shap')
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
from shap_utils import highlight_fingerprint_bits, get_important_shap_bits, find_bit
import numpy as np
from clogp_ground_truth import crippen_contrib_per_atom
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import io
from PIL import Image
import xgboost
import yaml
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from IPython.display import SVG, display
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# %% load config
import yaml
path = # absolute path to training/projects/kinase_inhibitors/231010-191518-bce_logits-y1-ll_fc
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

fpgen = Chem.GetMorganGenerator(radius=2, fpSize=1024)
X_train = np.array([fpgen.GetFingerprint(Chem.MolFromSmiles(graph.smiles)) for graph in tqdm(trainset)])
y_train = np.array([graph.y.item() for graph in trainset])
X_val = np.array([fpgen.GetFingerprint(Chem.MolFromSmiles(graph.smiles)) for graph in tqdm(valset)])
y_val = np.array([graph.y.item() for graph in valset])
X_test = np.array([fpgen.GetFingerprint(Chem.MolFromSmiles(graph.smiles)) for graph in tqdm(testset)])
y_test = np.array([graph.y.item() for graph in testset])
# %% train xgboost model 
model = xgboost.XGBClassifier(seed=42).fit(X_train, y_train)
preds_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, preds_train)
print(f"{train_accuracy=:.4f}")
train_f1_score = f1_score(y_train, preds_train)
print(f"{train_f1_score=:.4f}")
train_auroc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
print(f"{train_auroc=:.4f}")
print()
preds_val = model.predict(X_val)
val_accuracy = accuracy_score(y_val, preds_val)
print(f"{val_accuracy=:.4f}")
val_f1_score = f1_score(y_val, preds_val)
print(f"{val_f1_score=:.4f}")
val_auroc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"{val_auroc=:.4f}")
print()
preds_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, preds_test)
print(f"{test_accuracy=:.4f}")
test_f1_score = f1_score(y_test, preds_test)
print(f"{test_f1_score=:.4f}")
test_auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"{test_auroc=:.4f}")
# %% get explanations (shap values)
explainer_val = shap.Explainer(model, X_val)
shap_values_val = explainer_val(X_val)
# %% explain instance
data_instance = 9
print(f"pred={preds_val[data_instance]}, label={y_val[data_instance]}")
smi = valset[data_instance].smiles
mol = Chem.MolFromSmiles(smi)
ao = Chem.AdditionalOutput()
ao.CollectBitInfoMap()
fp = fpgen.GetFingerprint(mol, additionalOutput=ao)
bi = ao.GetBitInfoMap()
shap_values = shap_values_val[data_instance]
shap.plots.force(shap_values.base_values, shap_values.values, shap_values.data,
                 matplotlib=True, feature_names=[str(x) for x in range(len(shap_values.data))])

important_bits = get_important_shap_bits(shap_values.values, 10)
present_important_bits = [x for x in important_bits if x in bi.keys()]
important_bits = present_important_bits[0:2]
display(SVG(highlight_fingerprint_bits(mol, important_bits, bi, " | ".join([str(x) for x in important_bits]))))
display(Draw.DrawMorganBit(mol, important_bits[0], bi, useSVG=True))
# %% look at missing bits
128 in bi.keys()
missing_bit = 128
found_mol, found_bi = find_bit(missing_bit, fpgen, [graph.smiles for graph in valset])
Draw.DrawMorganBit(found_mol, missing_bit, found_bi, useSVG=True)

# %% draw example
smi = "OC1=CC=C(C2=NC3=CC=CC=C3C4=C5C(NC=C25)=NC=C4)C(Cl)=C1"
mol = Chem.MolFromSmiles(smi)
ao = Chem.AdditionalOutput()
ao.CollectBitInfoMap()
fp = fpgen.GetFingerprint(mol, additionalOutput=ao)
bi = ao.GetBitInfoMap()

pred = model.predict([fp])
explainer = shap.Explainer(model, X_train)
new_set = np.concatenate((X_val, np.array([fp])))
shap_values = explainer(new_set)
shap_values = shap_values[-1]


# %%
shap.plots.force(shap_values.base_values, shap_values.values, shap_values.data,
                 matplotlib=True, feature_names=[str(x) for x in range(len(shap_values.data))],
                 show=False)
plt.savefig('shapforce_3KCK.svg')
# shap.plots.waterfall(shap_values)
# %%
important_bits = get_important_shap_bits(shap_values.values, 10)
important_bits
present_important_bits = [x for x in important_bits if x in bi.keys()]
not_present_important_bits = [x for x in important_bits if x not in bi.keys()]
important_bits = present_important_bits[0:3]
display(SVG(highlight_fingerprint_bits(mol, important_bits, bi, " | ".join([str(x) for x in important_bits]))))
svg = (highlight_fingerprint_bits(mol, important_bits, bi, " | ".join([str(x) for x in important_bits])))
with open('shapmol_3KCK.svg', 'w') as f:
    f.write(svg)

# %%
smi_adv = "OC1=CC=C(C2=CC3=CC=CC=C3C4=C5C(CC=C25)=CC=C4)C(Cl)=C1"
mol = Chem.MolFromSmiles(smi_adv)
ao = Chem.AdditionalOutput()
ao.CollectBitInfoMap()
fp = fpgen.GetFingerprint(mol, additionalOutput=ao)
bi = ao.GetBitInfoMap()

pred = model.predict([fp])
explainer = shap.Explainer(model, X_train)
new_set = np.concatenate((X_val, np.array([fp])))
shap_values = explainer(new_set)
shap_values = shap_values[-1]
# %%
shap.plots.force(shap_values.base_values, shap_values.values, shap_values.data,
                 matplotlib=True, feature_names=[str(x) for x in range(len(shap_values.data))],
                 show=False)
plt.savefig('shapforce_3KCK_adv.svg')
# shap.plots.waterfall(shap_values)
important_bits = get_important_shap_bits(shap_values.values, 10)
important_bits
present_important_bits = [x for x in important_bits if x in bi.keys()]
not_present_important_bits = [x for x in important_bits if x not in bi.keys()]
important_bits = present_important_bits[0:3]
display(SVG(highlight_fingerprint_bits(mol, important_bits, bi, " | ".join([str(x) for x in important_bits]))))
svg = highlight_fingerprint_bits(mol, important_bits, bi, " | ".join([str(x) for x in important_bits]))
with open('shapmol_3KCK_adversarial.svg', 'w') as f:
    f.write(svg)
# display(Draw.DrawMorganBit(mol, not_present_important_bits[0], bi, useSVG=True))