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
from clogp_ground_truth import crippen_contrib_per_atom
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import io
from PIL import Image



# %% load_data
with open(f'data/240331-063122-gat-42_clogp_valset.pkl', "rb") as f:
    gat_val = pickle.load(f)
with open(f'data/240331-044944-gcn-42_clogp_valset.pkl', "rb") as f:
    gcn_val = pickle.load(f)
with open(f'data/240331-063122-gat-42_clogp_testset.pkl', "rb") as f:
    gat_test = pickle.load(f)
with open(f'data/240331-044944-gcn-42_clogp_testset.pkl', "rb") as f:
    gcn_test = pickle.load(f)
with open(f'data/240331-070735-gat-43_clogp_valset.pkl', "rb") as f:
    gat_val_43 = pickle.load(f)
with open(f'data/240331-052347-gcn-43_clogp_valset.pkl', "rb") as f:
    gcn_val_43 = pickle.load(f)

def best_predictions(data):
    errors = np.zeros(len(data))
    for k,v in data.items():
        errors[k] = abs(data[k]['label'] - data[k]['pred'])
    return np.argsort(errors)

best_gat_val = best_predictions(gat_val)
# best 10: [ 5325, 10823,  6631,  6634,  7901, 10788,   288,  6745,  1417, 6888]

# %%
for x in best_gat_val[:10]:
    smiles = gat_val[x]['smiles']
    weights = list(gat_val[x]['MyX'].values())
    try: 
        gt = crippen_contrib_per_atom(smiles)
    except:
        print(f"failed for {x}")
        continue
    vis.draw_with_weights(smiles, weights, 
                        f"images/ab__{x}", save_format='svg')
    vis.draw_colorbar(weights,f"images/ab__{x}_cb", save_format='svg')
    vis.draw_with_weights(smiles, gt, 
                        f"images/ab__{x}gt",
                        save_format='svg',
                        cmap=vis.blue_red)
    vis.draw_colorbar(gt,f"images/ab__{x}gt_cb",
                      cmap=vis.blue_red,
                      save_format='svg')

# %% get pred/ label
# chose 288
idx = 288
print(f"288\npred: {gat_val[idx]['pred']}\nlabel: {gat_val[idx]['label']}")
# %%
def get_pearson(dataset1, dataset2):
    p_data1 = []
    p_data2 = []
    p_between_datas = []
    failed = []
    assert len(dataset1) == len(dataset2)
    for i in trange(len(dataset1)):
        assert dataset1[i]['smiles'] == dataset2[i]['smiles']
        try:
            gt_dataset1 = crippen_contrib_per_atom(dataset1[i]['smiles'])
            gt_dataset2 = crippen_contrib_per_atom(dataset2[i]['smiles'])
        except:
            failed.append(i)
            continue
        my_dataset1 = list(dataset1[i]['MyX'].values())
        my_dataset2 = list(dataset2[i]['MyX'].values())
        p_data1.append(pearsonr(gt_dataset1, my_dataset1)[0])
        p_data2.append(pearsonr(gt_dataset2, my_dataset2)[0])
        p_between_datas.append(pearsonr(my_dataset1, my_dataset2)[0])
    print(f"{len(failed)=}")
    return p_data1, p_data2, p_between_datas

p_gat_val, p_gcn_val, p_between_val = get_pearson(gat_val, gcn_val)
p_gat_test, p_gcn_test, p_between_test = get_pearson(gat_test, gcn_test)
p_gat_val_43, p_gcn_val_43, p_between_val_43 = get_pearson(gat_val_43, gcn_val_43)

# %%
def draw_density(arr_1, arr_2, arr_3, title=False, fig_w=9, fig_h=5*9/7.5):
    cm_unit = 1/2.54
    fig, ax = plt.subplots(figsize=(fig_w*cm_unit, fig_h*cm_unit), dpi=200)
    # fig, ax = plt.subplots()
    sns.kdeplot(arr_1, cut=0, ax=ax, color='#099dd1', legend=True)
    sns.kdeplot(arr_2, cut=0, ax=ax, color='#333f48')
    sns.kdeplot(arr_3, cut=0, ax=ax, color='#E6B860', ls='--')
    custom_lines = [Line2D([0], [0], color='#099dd1'),
                    Line2D([0], [0], color='#333f48'),
                    Line2D([0], [0], color='#E6B860', ls='--')]
    
    ax.legend(custom_lines, ['GT vs GCN', 'GT vs GAT', "GCN vs GAT"])
    ax.set_xlabel('Pearson correlation coefficient (ρ)')
    ax.set_ylabel('Kernel density estimation')
    ax.tick_params(color='grey')
    ax.set_xlim((-1, 1))


    pad_left = 2.7*cm_unit #+ .4*cm_unit 
    pad_bottom = 2.8*cm_unit #+ .2*cm_unit 
    pad_right = .8*cm_unit
    pad_top = .8*cm_unit
    # w_space = 0.25*cm_unit
    # h_space = 0.5*cm_unit

    plt.subplots_adjust(0 + pad_left/fig_w, 
                        0 + pad_bottom/fig_h, 
                        1 - pad_right/fig_w, 
                        1 - pad_top/fig_h)#,
                        # 0 + 2*w_space/fig_w, 
                        # 0 + 2*h_space/fig_h)
    if not title:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='white')
        im = Image.open(buf)
        display(im) # type: ignore
        buf.close()
    else:
        plt.savefig(fname=title+'.svg')
        print("figure saved")
    plt.close()

# %%
draw_density(p_gcn_val_43, p_gat_val_43, p_between_val_43, title='images/c__clogp_gt_correlation_valset_43')
draw_density(p_gcn_val, p_gat_val, p_between_val, title='images/c__clogp_gt_correlation_valset')
# %%
draw_density(p_gcn_val_43, p_gat_val_43, p_between_val_43)
draw_density(p_gcn_val, p_gat_val, p_between_val)

# %%
import math
def get_mean(data):
    new_data = [x for x in data if not math.isnan(x)]
    return sum(new_data)/len(new_data)
print(f"{get_mean(p_gcn_val_43)=}")
print(f"{get_mean(p_gat_val_43)=}")
print(f"{get_mean(p_between_val_43)=}")
print(f"{get_mean(p_gcn_val)=}")
print(f"{get_mean(p_gat_val)=}")
print(f"{get_mean(p_between_val)=}")