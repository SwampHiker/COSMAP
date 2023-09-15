#**********************************************#
# The file with first stage neural network.    #
#   (fold prediction)                          #
#**********************************************#

import pytorch_lightning as pl
# import numpy as np
import torch
# import pickle as pkl
import torch.nn as nn
from torch.utils.data import DataLoader

from cosmap_fcn import CosMaP_FCN5_1D, CosMaP_FCN5
# import cosma_algebra as ca
import rectifier as rct
from blocks import FG_Pipeline, SG_Dataset

###  PARAMS  ###
batch_hp = 64
workers_hp = 8
learning_rate_hp = 1e-2 #2e-3
device_hp = 'cuda'
accel_hp = 'gpu'
train_path = './train.pkl'
valid_path = './valid.pkl'
outdir_path = './gen1_log'
epochs = 80 #always change...
#save_path = './sg_save.pkl'
train_crop = 64
#######
# encoder_path = './fg_log/first_gen64.pkl'  

################

torch.set_float32_matmul_precision("high")

class Gen1_Pipeline(FG_Pipeline):
    """GEN1: amino -> fold."""

    def __init__(self, train_pkl = None, valid_pkl = None):
        super().__init__()
        # self.body = rct.CosMaP_L_LML_1(40, 256)
        # self.head = rct.CosMaP_L_LM_Head(256, 1)
        # self.model = nn.Sequential(self.body, self.head)
        self.model = nn.Sequential(CosMaP_FCN5_1D(out1=128, input=40, output=256),
                                   rct.L_Mult_Layer(),
                                   CosMaP_FCN5(128, 128, 64, 64, 256, 1))
        self.loss = rct.SSIM_loss(1)

        self.train_set = None if train_pkl is None else SG_Dataset(train_pkl)
        self.valid_set = None if valid_pkl is None else SG_Dataset(valid_pkl)
        if not self.train_set is None:
            self.train_load = DataLoader(self.train_set, batch_hp, num_workers=workers_hp, pin_memory=False)
        if not self.valid_set is None:
            self.valid_load = DataLoader(self.valid_set, batch_hp, num_workers=workers_hp, pin_memory=False)


    def forward(self, amino):
        return self.model(amino.mT)
        
    def step(self, batch, batch_nb):
        cosma, amino = batch.values()
        return self.loss(self(amino), cosma)

def train(checkpoint = None, full_cont = True):
    pipeline = Gen1_Pipeline(train_path, valid_path) if checkpoint is None else Gen1_Pipeline.load_from_checkpoint(checkpoint, train_pkl=train_path, valid_pkl=valid_path)
    pipeline.set_crop(train_crop)
    print(f"Train set: {len(pipeline.train_set)}\nValid set: {len(pipeline.valid_set)}")
    callback = pl.callbacks.ModelCheckpoint(outdir_path, 'gen1_save')
    logger = pl.loggers.CSVLogger(outdir_path, 'logs', flush_logs_every_n_steps=50000)
    trainer = pl.Trainer(logger, callbacks=[callback], accelerator=accel_hp, max_epochs=epochs) #, track_grad_norm=2, detect_anomaly=True) #, limit_train_batches=0.10)
    if checkpoint is None or not full_cont:
        trainer.fit(pipeline)
    else:
        trainer.fit(pipeline, ckpt_path=checkpoint) #change in fg_pipeline.py ...
    # trainer.save_checkpoint(save_path)
    return pipeline

if __name__ == '__main__':
    # train('./sg_log/sg_save-100ep.ckpt', False)
    train('./gen1_log/gen1_save-20ep.ckpt')
