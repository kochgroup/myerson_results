# single run in project 
python train.py +dataset=clogp_smallzinc model.name=gat_3conv1fc lightning_trainer.max_epochs=2 project=testing_framework
# continue single run in project
python train.py -cp projects/testing_framework/231008-121157/.hydra/ resume_from_ckpt=projects/testing_framework/231008-121157/checkpoints/last.ckpt lightning_trainer.max_epochs=4
# multirun in project
python train.py +dataset=clogp_smallzinc model.name=gat_3conv1fc lightning_trainer.max_epochs=10 project=CPUvsGPU lightning_trainer.accelerator=cpu,gpu --multirun