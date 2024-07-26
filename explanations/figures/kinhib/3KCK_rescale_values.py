# %%
import sys, os
import pickle
# check for correct working directory 
assert os.getcwd().split('/')[-1] == "kinhib"
# add higher level directories to the path
path_to_root = os.getcwd()
while path_to_root.split('/')[-1] != 'myerson_results': path_to_root = '/'.join(path_to_root.split('/')[:-1])
sys.path.insert(0,path_to_root)
sys.path.insert(0, os.path.join(path_to_root, "training"))
from explanations.visualizations import draw_colorbar, draw_with_weights
# %%
with open("3KCK_adversarial2_myerson_values.pkl", "rb") as f:
    adv2 = pickle.load(f)
smi_adversarial2 = "OC1=CC=C(C2=NC3=CC=CC=C3C4=C5C(N(C)C=C25)=NC=C4)C(Cl)=C1"
draw_with_weights(smi_adversarial2, adv2)
# %%
with open("3KCK_adversarial_myerson_values.pkl", "rb") as f:
    adv = pickle.load(f)
smi_adversarial = "OC1=CC=C(C2=CC3=CC=CC=C3C4=C5C(CC=C25)=CC=C4)C(Cl)=C1"
draw_with_weights(smi_adversarial, adv)
# %%
with open("3KCK_myerson_values.pkl", "rb") as f:
    data = pickle.load(f)
smi = "OC1=CC=C(C2=NC3=CC=CC=C3C4=C5C(NC=C25)=NC=C4)C(Cl)=C1"

draw_with_weights(smi, data)

# %%
def rescale(myerson_dict: dict, lower=-1, upper=1) -> dict:
    min_ = min(myerson_dict.values())
    max_ = max(myerson_dict.values())

    d = {}
    for k, v in myerson_dict.items():
        x = ( (upper-lower)*(v - min_) / (max_-min_) ) + lower
        d.update({k: x})
    return d

# %%
draw_with_weights(smi, rescale(data))
draw_with_weights(smi_adversarial, rescale(adv))
draw_with_weights(smi_adversarial2, rescale(adv2))
draw_colorbar(rescale(adv2).values())
rescale(adv2)



