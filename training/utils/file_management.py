import os
import shutil
import yaml

def set_experiment_dir(experiment_dir, overwrite):
    """Creates experiment directory if it doesn't exist yet.

    Args:
        experiment_dir (str): The experiment directory.
    """
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    else:
        if overwrite == False:
            raise FileExistsError(f"{experiment_dir} exists and cmd argument --overwrite ist {overwrite}")

# def check_new_path

def copy_config(out_dir, run_name):
    src = os.path.join(out_dir, '.hydra', 'config.yaml')
    all_configs_path = os.path.join(out_dir, '..', 'all_configs')
    if not os.path.exists(all_configs_path):
        os.makedirs(all_configs_path)
    shutil.copy(src, os.path.join(all_configs_path, run_name+'.yaml'))

def append_hydra_args(out_dir):
    config_path = os.path.join(out_dir, '.hydra', 'config.yaml')
    hydra_config_path = os.path.join(out_dir, '.hydra', 'hydra.yaml')
    with open(hydra_config_path, 'r') as f:
        hydra_cfg = yaml.safe_load(f)
    hydra_dir_args = {'hydra': {'run': hydra_cfg['hydra']['run'],
                                'sweep': hydra_cfg['hydra']['sweep']}}
    with open(config_path, 'a') as f:
        yaml.safe_dump(hydra_dir_args, f)


