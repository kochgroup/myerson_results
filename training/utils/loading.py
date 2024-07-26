
from utils.register import (loss_fn_dict, metric_dicts, optim_dict, 
                            model_dict, dataset_dict)
from models import *
from datasets import *

def load_loss(key:str):
    return loss_fn_dict[key]
    
def load_optimizer(key:str):
    return optim_dict[key]

def load_model(key:str):
    return model_dict[key]

def load_dataset(key:str):
    return dataset_dict[key]

def default_metrics(task_type:str):
    return metric_dicts[task_type]

def test_metrics(task_type:str, use_additional_metrics:bool):
    if use_additional_metrics:
        return {**metric_dicts[task_type], **metric_dicts[task_type+"_test"]}
    else:
        return {metric_dicts[task_type]}