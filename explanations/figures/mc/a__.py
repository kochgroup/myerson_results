# %%
import os
project_root = # absolute path to myerson_results repo

working_dir = os.path.join(project_root, 'explanations/figures/mc')
os.chdir(working_dir)

import yaml
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

with open('data/mc_my.yaml', 'r') as f:
    mc_rand = yaml.safe_load(f)
with open('data/mc_42_my.yaml', 'r') as f:
    mc_42 = yaml.safe_load(f)
with open('data/mc_43_my.yaml', 'r') as f:
    mc_43 = yaml.safe_load(f)
with open('data/mc_44_my.yaml', 'r') as f:
    mc_44 = yaml.safe_load(f)
with open('data/mc_69_my.yaml', 'r') as f:
    mc_69 = yaml.safe_load(f)
with open('data/mc_420_my.yaml', 'r') as f:
    mc_420 = yaml.safe_load(f)
with open('data/exact_my.yaml', 'r') as f:
    exact = yaml.safe_load(f)
with open('data/mc_my_manystep_pred.yaml', 'r') as f:
    mc_manystep = yaml.safe_load(f)


# %% get dataframes for sns
def get_graph_error_vs_graph_size(data):
    size = []
    err = []
    rel_err = []
    samples = []
    mol_idx = []
    for size_key in tqdm(data.keys()):
        for idx_key in data[size_key].keys():
            for my_samples in ['my_10','my_100','my_500','my_1000','my_5000']:
                size.append(size_key)
                mol_idx.append(idx_key)
                samples.append(my_samples[3:])
                sum_my = sum(data[size_key][idx_key][my_samples].values())
                pred = data[size_key][idx_key]['pred']
                error = abs(sum_my - pred)
                rel_error = error / pred
                err.append(error)
                rel_err.append(rel_error)
    return pd.DataFrame({'Graph size || Number heavy atoms': size,
                         'Absolute error Σ(Myerson)': err,
                         'Relative error Σ(Myerson)': rel_err,
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         })
def get_graph_error_vs_graph_size_seeded(datalist: list):
    size = []
    err = []
    rel_err = []
    samples = []
    mol_idx = []
    for data in tqdm(datalist): 
        for size_key in (data.keys()):
            for idx_key in data[size_key].keys():
                for my_samples in ['my_10','my_100','my_500','my_1000','my_5000']:
                    size.append(size_key)
                    mol_idx.append(idx_key)
                    samples.append(my_samples[3:])
                    sum_my = sum(data[size_key][idx_key][my_samples].values())
                    pred = data[size_key][idx_key]['pred']
                    error = abs(sum_my - pred)
                    rel_error = error / pred
                    err.append(error)
                    rel_err.append(rel_error)
                break # just take the first example
    return pd.DataFrame({'Graph size || Number heavy atoms': size,
                         'Absolute error Σ(Myerson)': err,
                         'Relative error Σ(Myerson)': rel_err,
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         })
def get_time_vs_size(data_mc, data_exact):
    size = []
    samples = []
    mol_idx = []
    times = []
    assert len(data_exact) <= len(data_mc)
    for size_key in tqdm(data_exact.keys()):
        for idx_key in data_exact[size_key].keys():
            for my_samples in ['my_10','my_100','my_500','my_1000','my_5000', 'exact']:
                size.append(size_key)
                mol_idx.append(idx_key)
                if my_samples == 'exact':
                    samples.append('exact')
                    times.append(data_exact[size_key][idx_key]['time'])
                else:
                    samples.append(my_samples[3:])
                    times.append(data_mc[size_key][idx_key][f'{my_samples}_time'])
    for size_key in [x for x in data_mc.keys() if x not in data_exact.keys()]:
        for idx_key in data_mc[size_key].keys():
            for my_samples in ['my_10','my_100','my_500','my_1000','my_5000', 'exact']:
                size.append(size_key)
                mol_idx.append(idx_key)
                if my_samples == 'exact':
                    samples.append('exact')
                    times.append(float('nan'))
                else:
                    samples.append(my_samples[3:])
                    times.append(data_mc[size_key][idx_key][f'{my_samples}_time'])
    return pd.DataFrame({'Graph size || Number heavy atoms': size,
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         'Time / s': times,
                         })
def _get_mean_atom_err(my_mc, my_exact):
    my_mc = np.array(list(my_mc.values()))
    my_exact = np.array(list(my_exact.values()))
    res =  np.sum(np.abs(my_mc - my_exact))/len(my_mc)
    rel_res =  np.sum(np.abs(my_mc - my_exact)/np.abs(my_exact))/len(my_mc)
    return res, rel_res
def _get_max_atom_err(my_mc, my_exact):
    my_mc = np.array(list(my_mc.values()))
    my_exact = np.array(list(my_exact.values()))
    res = np.max(np.abs(my_mc - my_exact))
    rel_res = np.max(np.abs(my_mc - my_exact)/np.abs(my_exact))
    return res, rel_res

def get_atom_error_vs_graph_size(data_mc, data_exact):
    size = []
    samples = []
    mol_idx = []
    mean_atom_err = []
    max_atom_err = []
    rel_mean_atom_err = []
    rel_max_atom_err = []
    assert len(data_exact) <= len(data_mc)
    for size_key in tqdm(data_exact.keys()):
        for idx_key in data_exact[size_key].keys():
            for my_samples in ['my_10','my_100','my_500','my_1000','my_5000']:
                size.append(size_key)
                mol_idx.append(idx_key)
                samples.append(my_samples[3:])
                mean_err, rel_mean_err = _get_mean_atom_err(data_mc[size_key][idx_key][my_samples],
                                                   data_exact[size_key][idx_key]['my_values'])
                max_err, rel_max_err = _get_max_atom_err(data_mc[size_key][idx_key][my_samples],
                                                   data_exact[size_key][idx_key]['my_values'])
                mean_atom_err.append(mean_err)
                rel_mean_atom_err.append(rel_mean_err)
                max_atom_err.append(max_err)
                rel_max_atom_err.append(rel_max_err)
    return pd.DataFrame({'Graph size || Number heavy atoms': size,
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         'Mean node error': mean_atom_err,
                         'Relative mean node error': rel_mean_atom_err,
                         'Max node error': max_atom_err,
                         'Relative max node error': rel_max_atom_err,
                         })
def get_atom_error_vs_graph_size_seeded(mc_datalist, data_exact):
    size = []
    samples = []
    mol_idx = []
    mean_atom_err = []
    max_atom_err = []
    rel_mean_atom_err = []
    rel_max_atom_err = []
    for data in tqdm(mc_datalist): 
        for size_key in (data_exact.keys()):
            for idx_key in data[size_key].keys():
                for my_samples in ['my_10','my_100','my_500','my_1000','my_5000']:
                    size.append(size_key)
                    mol_idx.append(idx_key)
                    samples.append(my_samples[3:])
                    mean_err, rel_mean_err = (_get_mean_atom_err(data[size_key][idx_key][my_samples],
                                                   data_exact[size_key][idx_key]['my_values']))
                    max_err, rel_max_err = (_get_max_atom_err(data[size_key][idx_key][my_samples],
                                                    data_exact[size_key][idx_key]['my_values']))
                    mean_atom_err.append(mean_err)
                    rel_mean_atom_err.append(rel_mean_err)
                    max_atom_err.append(max_err)
                    rel_max_atom_err.append(rel_max_err)
                break # just take the first example
    return pd.DataFrame({'Graph size || Number heavy atoms': size,
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         'Mean node error': mean_atom_err,
                         'Relative mean node error': rel_mean_atom_err,
                         'Max node error': max_atom_err,
                         'Relative max node error': rel_max_atom_err,
                         })
def get_graph_error_vs_step_size(data):
    samples = []
    err = []
    rel_err = []
    mol_idx = []
    for idx_key in tqdm(data.keys()):
        for sample_size in data[idx_key].keys():
            if sample_size in ['smiles', 'pred']:
                continue
            mol_idx.append(idx_key)
            sum_my = sum(data[idx_key][sample_size].values())
            pred = data[idx_key]['pred']
            error = abs(sum_my - pred)
            rel_error = error / pred
            err.append(error)
            rel_err.append(rel_error)
            samples.append(sample_size)
    return pd.DataFrame({
                         'Samples': samples,
                         'mol_idx': mol_idx,
                         'Absolute error Σ(Myerson)': err,
                         'Relative error Σ(Myerson)': rel_err,
                         })
            



df = get_graph_error_vs_graph_size(mc_rand)
seed_df = get_graph_error_vs_graph_size_seeded([mc_42, mc_43, mc_44, mc_69, mc_420])
time_df = get_time_vs_size(mc_rand, exact)
atom_err_df = get_atom_error_vs_graph_size(mc_rand, exact)
seed_atom_err_df = get_atom_error_vs_graph_size_seeded([mc_42, mc_43, mc_44, mc_69, mc_420], exact)
manystep_df = get_graph_error_vs_step_size(mc_manystep)

# %%
safe_figures = True
# sns.pointplot(df, x='size', y='err', hue='samples', linestyle='none')
# PLOT 1: ERROR MYERSON SUM VS GRAPH SIZE ON 10 EXAMPLES EACH 
if safe_figures: 
    plot = sns.lineplot(df, x='Graph size || Number heavy atoms', y='Absolute error Σ(Myerson)', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/a__myerr_size.svg')
    plt.close()
else:
    sns.lineplot(df, x='Graph size || Number heavy atoms', y='Absolute error Σ(Myerson)', hue='Samples',)
    plt.show()
    sns.lineplot(df, x='Graph size || Number heavy atoms', y='Relative error Σ(Myerson)', hue='Samples',)
    plt.show()

# PLOT 2: [SI] ERROR MYERSON SUM VS GRAPH SIZE ON 1 EXAMPLE FOR 5 DIFFERENT SEEDS
if safe_figures: 
    plot = sns.lineplot(seed_df, x='Graph size || Number heavy atoms', y='Absolute error Σ(Myerson)', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__myerrseed_size.svg')
    plt.close()
else:
    sns.lineplot(seed_df, x='Graph size || Number heavy atoms', y='Absolute error Σ(Myerson)', hue='Samples',)
    plt.show()
    sns.lineplot(seed_df, x='Graph size || Number heavy atoms', y='Relative error Σ(Myerson)', hue='Samples',)
    plt.show()
# PLOT 3: PER-NODE ERROR VS GRAPH SIZE ON 10 EXAMPLES EACH
if safe_figures: 
    plot = sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Mean node error', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__meannodeerr_size.svg')
    plt.close()
    plot = sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Max node error', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__maxnodeerr_size.svg')
    plt.close()
else:
    sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Mean node error', hue='Samples',)
    plt.show()
    sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Max node error', hue='Samples',)
    plt.show()
    sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Relative mean node error', hue='Samples',)
    plt.show()
    sns.lineplot(atom_err_df, x='Graph size || Number heavy atoms', y='Relative max node error', hue='Samples',)
    plt.show()
# PLOT 4: PER-NODE ERROR VS GRAPH SIZE ON 1 EXAMPLE FOR 5 DIFFERENT SEEDS
if safe_figures: 
    plot = sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Mean node error', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__meannodeerrseed_size.svg')
    plt.close()
    plot = sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Max node error', hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__maxnodeerrseed_size.svg')
    plt.close()
else:
    sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Mean node error', hue='Samples',)
    plt.show()
    sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Max node error', hue='Samples',)
    plt.show()
    sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Relative mean node error', hue='Samples',)
    plt.show()
    sns.lineplot(seed_atom_err_df, x='Graph size || Number heavy atoms', y='Relative max node error', hue='Samples',)
    plt.show()
# PLOT 5: CALCULATION TIME VS GRAPH SIZE 
if safe_figures: 
    plot = sns.lineplot(time_df, x='Graph size || Number heavy atoms', y='Time / s', hue='Samples')
    plt.ylim(top=102, bottom=-2)
    fig = plot.get_figure()
    fig.savefig('images/b__time_size.svg')
    plt.close()
    plot = sns.lineplot(time_df, x='Graph size || Number heavy atoms', y='Time / s', hue='Samples')
    plt.yscale('log')
    plt.ylim(top=5000, bottom=-2)
    fig = plot.get_figure()
    fig.savefig('images/si__logtime_size.svg')
    plt.close()
else:
    sns.lineplot(time_df, x='Graph size || Number heavy atoms', y='Time / s', hue='Samples')
    plt.ylim(top=102, bottom=-2)
    plt.show()
        # LOG SCALE
    sns.lineplot(time_df, x='Graph size || Number heavy atoms', y='Time / s', hue='Samples')
    plt.yscale('log')
    plt.ylim(top=5000, bottom=-2)
    plt.show()
# PLOT 6: SAMPLES VS ERROR, 25 ATOMS
if safe_figures: 
    plot = sns.lineplot(manystep_df, x='Samples', y='Absolute error Σ(Myerson)',)# hue='Samples',)
    fig = plot.get_figure()
    fig.savefig('images/si__err_samples.svg')
    plt.close()
else:
    sns.lineplot(manystep_df, x='Samples', y='Absolute error Σ(Myerson)',)# hue='Samples',)
    plt.show()
    sns.lineplot(manystep_df, x='Samples', y='Relative error Σ(Myerson)',)# hue='Samples',)
    plt.show()
