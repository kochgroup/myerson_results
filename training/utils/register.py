# metrics
import torchmetrics
from utils.metrics import (binary_auroc, ModKendallRankCorrCoef, ModBinaryAccuracy,
                           ModBinaryAUROC, ModBinaryF1Score)
f_metric_dict = {
    'mse': torchmetrics.functional.mean_squared_error,
    'mae': torchmetrics.functional.mean_absolute_error,
    'pearson_r': torchmetrics.functional.pearson_corrcoef,
    'kendall_tau': torchmetrics.functional.kendall_rank_corrcoef,
    'accuracy': torchmetrics.functional.classification.binary_accuracy,
    'binary_auroc': binary_auroc,
}
regression_metric_dict = {
    'mse': torchmetrics.MeanSquaredError,
    'mae': torchmetrics.MeanAbsoluteError,
    'pearson_r': torchmetrics.PearsonCorrCoef,
    'r2': torchmetrics.R2Score,
}
regression_metric_dict_test = {
    'kendall_tau': ModKendallRankCorrCoef,
}
binary_classification_metric_dict = {
    'accuracy': ModBinaryAccuracy,
    'auroc': ModBinaryAUROC,
    'f1score': ModBinaryF1Score

}
binary_classification_metric_dict_test = {

}
multiclass_classification_metric_dict = {

}
multiclass_classification_metric_dict_test = {

}
multilabel_classification_metric_dict = {

}
multilabel_classification_metric_dict_test = {

}

metric_dicts = {
    'regression': regression_metric_dict,
    'regression_test': regression_metric_dict_test,
    'binary_classification': binary_classification_metric_dict,
    'binary_classification_test': binary_classification_metric_dict_test,
    'multiclass_classification': multiclass_classification_metric_dict,
    'multiclass_classification': multiclass_classification_metric_dict_test,
    'multilabel_classification': multilabel_classification_metric_dict,
    'multilabel_classification': multilabel_classification_metric_dict_test
}

# loss
import torch.nn.functional as F
loss_fn_dict = {
    'mse': F.mse_loss,
    'bce': F.binary_cross_entropy,
    'bce_logits': F.binary_cross_entropy_with_logits,
}

# optimizer
from torch import optim
optim_dict = {
    'adam': optim.Adam
}

# datasets
dataset_dict = {}
def register_model(key: str, module: any = None):
    return register_base(model_dict, key, module)

# models
model_dict = {}
def register_dataset(key: str, module: any = None):
    return register_base(dataset_dict, key, module)
    
# register decorator
def register_base(mapping: dict[str, any], key: str,
                  module: any = None) -> [None, callable]:
    r"""Base function for registering a module, code copied from GraphGym.

    Args:
        mapping (dict): Python dictionary to register the module.
            hosting all the registered modules
        key (str): The name of the module.
        module (any, optional): The module. If set to :obj:`None`, will return
            a decorator to register a module.
    """
    if module is not None:
        if key in mapping:
            raise KeyError(f"Module with '{key}' already defined")
        mapping[key] = module
        return

    # Other-wise, use it as a decorator:
    def bounded_register(module):
        register_base(mapping, key, module)
        return module

    return bounded_register



