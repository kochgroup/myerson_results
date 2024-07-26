import pytorch_lightning as pl
from .loading import load_loss, load_optimizer, default_metrics, test_metrics
import torchmetrics # TODO: move out to register / loading 
import os
import torch

class LitModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # self.save_hyperparameters()
        self.cfg = cfg
        self.model = model
        self.loss_fun = load_loss(cfg.lightning_module.loss_fun)
        self.terminal_width = os.get_terminal_size()[0]

        if cfg.fit:
            metric_dict = default_metrics(self.cfg.lightning_module.task_type)
            metrics = torchmetrics.MetricCollection({k: v() for k,v in metric_dict.items()})
            self.train_metrics = metrics.clone(prefix='train_')
            self.val_metrics = metrics.clone(prefix='val_')
            self.train_loss = []
            self.val_loss = []
        if cfg.test:
            metric_dict = test_metrics(self.cfg.lightning_module.task_type, cfg.testing_with_additional_metrics)
            metrics = torchmetrics.MetricCollection({k: v() for k,v in metric_dict.items()})
            self.test_metrics = metrics.clone(prefix='test_')
            self.test_loss = []



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        pred = self.model(batch.x, batch.edge_index, batch.batch, *args, **kwargs)
        label = batch.y
        loss = self.loss_fun(pred, label)
        self.train_loss.append(loss)
        out_metrics = self.train_metrics(pred, label)
        self.log_dict({k+'_step': v for k,v in out_metrics.items()})
        self.log('train_loss_step', loss)
        return loss

    def on_train_epoch_end(self):
        out_metrics = self.train_metrics.compute()
        self.log_dict(out_metrics)
        epoch_loss = torch.mean(torch.stack(self.train_loss))
        self.log('train_loss', epoch_loss)
        epoch_loss = {'train_loss': round(epoch_loss.item(), 4)}
        out_metrics = {k: round(v.item(), 4) for k, v in out_metrics.items()}
        out_metrics.update(epoch_loss)
        msg = (f"{' '*len('Epoch '+str(self.current_epoch))} Train metrics: {out_metrics}")
        print(f"\r{msg:<{self.terminal_width}}") # pad when overwriting progressbar
        self.train_loss = []
        self.train_metrics.reset()
        
    def validation_step(self, batch, *args, **kwargs):
        # only log validation per epoch
        pred = self.model(batch.x, batch.edge_index, batch.batch, *args, **kwargs)
        label = batch.y
        loss = self.loss_fun(pred, label)
        self.val_loss.append(loss)
        self.val_metrics.update(pred, label)
        return loss

    def on_validation_epoch_end(self):
        out_metrics = self.val_metrics.compute()
        self.log_dict(out_metrics)
        epoch_loss = torch.mean(torch.stack(self.val_loss))
        self.log('val_loss', epoch_loss) 
        epoch_loss = {'val_loss': round(epoch_loss.item(), 4)}
        out_metrics = {k: round(v.item(),4) for k, v in out_metrics.items()}
        out_metrics.update(epoch_loss)
        msg = (f"Epoch {self.current_epoch} - Val metrics: {out_metrics}")
        print(f"\r{msg:<{self.terminal_width}}") # pad when overwriting progressbar
        self.val_loss = []
        self.val_metrics.reset()

    def test_step(self, batch, *args, **kwargs):
        pred = self.model(batch.x, batch.edge_index, batch.batch, *args, **kwargs)
        label = batch.y
        loss = self.loss_fun(pred, label)
        self.test_loss.append(loss)
        self.test_metrics.update(pred, label)
        return loss

    def on_test_epoch_end(self):
        out_metrics = self.test_metrics.compute()
        self.log_dict(out_metrics)
        epoch_loss = torch.mean(torch.stack(self.test_loss))
        self.log('test_loss', epoch_loss) 
        self.test_loss = []
        self.test_metrics.reset()


    def configure_optimizers(self):
        optimizer =  load_optimizer(self.cfg.lightning_module.optimizer)
        return optimizer(self.parameters(), 
                         lr=self.cfg.lightning_module.base_lr, 
                         weight_decay=self.cfg.lightning_module.weight_decay)

from pytorch_lightning import Callback
import datetime, time
import json

class PrintOnFitEnd(Callback):
    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir
        self.train_start_time = None
        self.terminal_width = os.get_terminal_size()[0]
    def on_fit_start(self, trainer, pl_module): 
        self.train_start_time = time.time()
    def on_fit_end(self, trainer, pl_module):
        # removing second printing of progress bar
        print(f"\033[F{'':<{self.terminal_width}}", end='')

        # print and save end metrics
        training_time = datetime.timedelta(seconds=(time.time() - self.train_start_time))
        epoch_time = training_time/pl_module.current_epoch
        step_time = training_time/pl_module.global_step
        time_metrics = ({'training_time': str(training_time),
                         'time/step': str(step_time),
                         'time/epoch': str(epoch_time)})
        val_metrics = {k: round(v.item(),4) for k, v in trainer.callback_metrics.items() if "val_" in k}
        train_metrics = {k: round(v.item(),4) for k, v in trainer.callback_metrics.items() if ("train_" in k and "_step" not in k)}

        end_print_metrics = {}
        end_print_metrics.update({k[4:]: v for k,v in val_metrics.items()})
        end_print_metrics.update({'--- ': '---'})
        end_print_metrics.update(time_metrics)
        for k,v in end_print_metrics.items():
            print(f"{k:>{max([len(x) for x in end_print_metrics.keys()])}}: {v}")
        
        epoch_metrics = {"val_metrics": val_metrics, "train_metrics": train_metrics,
                         "time_metrics": time_metrics}
        json.dump(epoch_metrics, open(self.out_dir+'/fit_stats.json', "w"))
        
class PrintOnTestEnd(Callback):
    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir
        self.terminal_width = os.get_terminal_size()[0]
    def on_test_end(self, trainer, pl_module):
        test_metrics = {k: round(v.item(),4) for k, v in trainer.callback_metrics.items() if ("test_" in k)}
        json.dump(test_metrics, open(self.out_dir+'/test_stats.json', "w"))
