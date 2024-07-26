from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    # imports (tab completion in hydra is slow otherwise)
    import os
    import logging
    import torch
    import wandb
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from torch_geometric import seed_everything
    from torch_geometric.data.lightning import LightningDataset

    from utils.loading import load_dataset, load_model
    from utils.shuffle import shuffle_dataset
    from utils.lightning import LitModule, PrintOnFitEnd, PrintOnTestEnd
    from utils.splitting import dataset_split
    from utils.dimensions import infer_dims
    from utils.file_management import copy_config, append_hydra_args

    # configure outputs
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    append_hydra_args(out_dir) # in case of reruns
    copy_config(out_dir, cfg.run)
    # logging
    hydra_log = logging.getLogger(__name__)
    loggers = []
    if cfg.logging.wandb:
        wandb_dir = os.path.join(out_dir, '..')
        loggers.append(WandbLogger(**cfg.wandb_kwargs))
    if cfg.logging.tensorboard:
        loggers.append(TensorBoardLogger(version='tensorboard', save_dir=out_dir))
    if cfg.logging.csv:
        loggers.append(CSVLogger(version='csv', save_dir=out_dir))

    # seed and precision
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.torch.num_threads)
    torch.set_float32_matmul_precision(cfg.torch.float32_precision)

    # dataset
    dataset_path = os.path.join(cfg.dataset.dir, cfg.dataset.name)
    dataset_class = load_dataset(cfg.dataset.name)
    dataset = dataset_class(root=dataset_path, **cfg.dataset)
    dataset = shuffle_dataset(dataset, seed=cfg.dataset.shuffle_seed)
    split_fn = dataset_split(cfg.dataset.split_mode)
    trainset, valset, testset = split_fn(dataset, cfg.dataset.split)
    datamodule = LightningDataset(trainset, valset, testset, 
                                  num_workers=cfg.torch.num_workers,
                                  batch_size=cfg.batch_size)

    # model
    model_class = load_model(cfg.model.name)
    model_kwargs = {}
    if 'dim_in' not in cfg.model.keys(): 
        dim_in, _ = infer_dims(dataset[0], cfg.dataset.data_type)
        model_kwargs['dim_in'] = dim_in
    if 'dim_out' not in cfg.model.keys(): 
        _, dim_out = infer_dims(dataset[0], cfg.dataset.data_type)
        model_kwargs['dim_out'] = dim_out
    model_kwargs = {**model_kwargs, **cfg.model}
    model = model_class(**model_kwargs)
    pl_model = LitModule(model, cfg)

    # training & testing
    callbacks = []
    checkpoint_cb = ModelCheckpoint(dirpath=out_dir+'/checkpoints', 
                                          save_last=True, 
                                          **cfg.checkpointing)
    callbacks.append(checkpoint_cb)
    early_stopping_cb = EarlyStopping(**cfg.early_stopping)
    if cfg.stop_early: callbacks.append(early_stopping_cb)
    fit_end_cb = PrintOnFitEnd(out_dir)
    callbacks.append(fit_end_cb)
    test_end_cb = PrintOnTestEnd(out_dir)
    callbacks.append(test_end_cb)

    lightning_kwargs = cfg.lightning_trainer
    trainer = pl.Trainer(callbacks=callbacks, 
                         logger=loggers,
                         **lightning_kwargs) 
    if cfg.fit:
        trainer.fit(pl_model, datamodule=datamodule, ckpt_path=cfg.resume_from_ckpt)
    if cfg.test:
        trainer.test(pl_model, datamodule=datamodule, ckpt_path=cfg.resume_from_ckpt)

    wandb.finish()

if __name__ == '__main__':
    main()