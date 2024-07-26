def rename_state_dict_keys(state_dict):
    """Change name of model parameter  tensors, e.g., `model.fc1.bias` --> 
    `fc1.bias` to use with pure pytorch instead of lightning."""
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k in state_dict:
        new_key = '.'.join(k.split('.')[1:])
        new_dict.update({new_key: state_dict[k]})
    return new_dict