import torch

def shuffle_dataset(dataset, seed):
    """Reproducably shuffle a dataset.

    Args:
        dataset (PyG Dataset): The dataset.
        seed (int): The seed.

    Returns:
        PyG Dataset: The shuffled dataset.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm=torch.randperm(len(dataset), generator=gen)
    return dataset.index_select(perm)
