# A framework for deep machine learning experiments

## Using Hydra, PyTorch Lightning and WandB (optional)

A framework to quickly and easily run different machine learning experiments using PyTorch Lightning and Hydra. Experiments can optionally be logged and analyzed with Weights  Biases.
Example commands can be found in `run_commands.sh`.

The idea is to copy the folder for a larger project (e.g. a paper) and register datasets and models. These can then be controlled by the config-file or hydra flags. Within the larger scheme, sub-projects will be saved in the project folder, so they can be easily compared using Weights & Biases or Tensorboard.

This allows for easy testing of different hyperparameters or models.

## Features to integrate

- [ ] Add support for runs with slurm / hpc.
- [ ] Resume run within same run directory?
- [ ] Add metrics for classification (binary, multiclass, multilabel)
- [ ] Hyperparameter optimization (Ax, Nevergrad, ...)
- [ ] ...
