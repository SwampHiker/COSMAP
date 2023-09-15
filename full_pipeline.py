#**********************************************#
# The file with second stage neural network.   #
#   (interaction prediction)                   #
#**********************************************#

import pytorch_lightning as pl
import numpy as np
import torch
import pickle as pkl
import torch.nn as nn
from torch.utils.data import DataLoader

from cosmap_fcn import CosMaP_FCN5_1D, CosMaP_FCN5
import cosma_algebra as ca
import rectifier as rct
from blocks import FG_Pipeline, Gen3_Dataset

# ##  PARAMS  ###
batch_hp = 64
workers_hp = 8
learning_rate_hp = 1e-2  # 2e-3
device_hp = 'cuda'
accel_hp = 'gpu'
train_path = './train_dim.pkl'
valid_path = './valid_dim.pkl'
outdir_path = './full_log/'
epochs = 100  # always change...
# save_path = './sg_save.pkl'
train_crop = 64
#######
# gen3_path = './gen3_log/gen3_back.pkl'
gen3_path = './gen1_log/gen1_pipe.pkl'


################


class Full_Net(nn.Module):  # nah, that's quite good actuallly
    """Implements gen3 training - predicting cosmas AND cv vectors
       SIMULTANEOUSLY!"""
    def __init__(self, gen1_model):
        super().__init__()
        # self.cs_encoder = gen2_model.cs_encoder  # added with recent GEN2
        self.cs_encoder = None
        # self.am_encoder = gen2_model.am_encoder
        # self.am_decoder = gen2_model.am_decoder
        self.am_encoder = gen1_model[0]
        self.am_decoder = gen1_model[2]

        # Freezing
        # for param in self.cs_encoder.parameters():
        #     param.requires_grad = False
        for param in self.am_encoder.parameters():
            param.requires_grad = False
        for param in self.am_decoder.parameters():
            param.requires_grad = False

        self.recoder = CosMaP_FCN5_1D(256+64, 256+256, 256+512, 256+512,
                                      256+40, 256)
        self.reformer = CosMaP_FCN5(256, 256, 512, 512, 256, 256 * 3)
        # self.reformer = CosMaP_FCN5(64, 128, 256, 512, 256, 256 * 3)
        self.mult = rct.A_Mult_Layer()

    def encode_cosma(self, cosma, amino):  # recent GEN2
        with torch.no_grad():
            return self.cs_encoder(cosma, amino.mT)

    def encode_cosma_grad(self, cosma, amino):  # recent GEN2
        return self.cs_encoder(cosma, amino.mT)

    def encode_amino(self, amino):
        with torch.no_grad():
            return self.am_encoder(amino.mT)

    def forward(self, x):
        amino1, code256_1, amino2, code256_2 = x
        input1 = torch.cat([amino1.mT, code256_1], 1)
        input2 = torch.cat([amino2.mT, code256_2], 1)
        outs = self.reformer(self.mult(self.recoder(input1),
                                       self.recoder(input2)))
        return torch.cat([self.am_decoder(outs[:, :256, ...]),
                          self.am_decoder(outs[:, 256:512, ...]),
                          self.am_decoder(outs[:, 512:, ...])], 1)


class Full_Dataset(Gen3_Dataset):
    """Dataset, which loads complexes [FULL] from pickles."""

    def __init__(self, file_pkl):
        super().__init__(file_pkl)

    def __getitem__(self, item, shuffle=True, ignore_crop=False):
        chain_dict = self.load_item(item)
        pmtrx_crop1, pmtrx_crop2 = chain_dict['pmtrx']
        amino_crop1, amino_crop2 = chain_dict['amino']

        coord1, coord2 = chain_dict['coord']
        cv = ca.get_cv(coord1, coord2)

        if not ignore_crop and self.crop is not None:
            i = np.random.choice(range(chain_dict['crop'][0] - self.crop[0]))
            j = np.random.choice(range(chain_dict['crop'][1] - self.crop[1]))
            pmtrx_crop1 = pmtrx_crop1[:, i:i+self.crop[0]]
            pmtrx_crop2 = pmtrx_crop2[:, j:j+self.crop[1]]
            amino_crop1 = amino_crop1[i:i+self.crop[0]+1]
            amino_crop2 = amino_crop2[j:j+self.crop[1]+1]

        # tmtrx1 = ca.get_tmtrx(pmtrx_crop1)
        # tmtrx2 = ca.get_tmtrx(pmtrx_crop2)

        cv_crop1 = pmtrx_crop1.T @ cv
        cv_crop2 = pmtrx_crop2.T @ (-cv)

        pmtrx_crop1 = np.expand_dims(pmtrx_crop1.astype(np.float32), 0)
        pmtrx_crop2 = np.expand_dims(pmtrx_crop2.astype(np.float32), 0)
        cv_crop1 = np.expand_dims(cv_crop1.astype(np.float32), 0)
        cv_crop2 = np.expand_dims(cv_crop2.astype(np.float32), 0)

        amino_crop1 = ca.amino_list_to_array_stack(amino_crop1,
                                                   False).astype(np.float32)
        amino_crop2 = ca.amino_list_to_array_stack(amino_crop2,
                                                   False).astype(np.float32)

        # return {'pmtrx1': pmtrx_crop1, 'pmtrx2': pmtrx_crop2,
        #         'amino1': amino_crop1, 'amino2': amino_crop2}

        if shuffle and np.random.rand() < 0.5:
            return (pmtrx_crop2, pmtrx_crop1,
                    amino_crop2, amino_crop1,
                    cv_crop2, cv_crop1)
        else:
            return (pmtrx_crop1, pmtrx_crop2,
                    amino_crop1, amino_crop2,
                    cv_crop1, cv_crop2)  # that should work...


class Full_Pipeline(FG_Pipeline):
    """GEN3: amino -> code256 -> *recoding, multiplying reforming*
                                            -> COSMA_Q, CV1, CV2."""

    def __init__(self, train_pkl=None, valid_pkl=None):
        super().__init__()

        with open(gen3_path, 'rb') as f:
            gen3_model = pkl.load(f).model

        self.model = Full_Net(gen3_model)
        self.loss = rct.SSIM_loss(3)

        self.train_set = None if train_pkl is None else Full_Dataset(train_pkl)
        self.valid_set = None if valid_pkl is None else Full_Dataset(valid_pkl)
        if self.train_set is not None:
            self.train_load = DataLoader(self.train_set, batch_hp,
                                         num_workers=workers_hp,
                                         pin_memory=False)
        if self.valid_set is not None:
            self.valid_load = DataLoader(self.valid_set, batch_hp,
                                         num_workers=workers_hp,
                                         pin_memory=False)

    def prepare_true_tensor(self, pmtrx1, pmtrx2, cv1, cv2):
        m, n = cv1.size(2), cv2.size(2)
        return torch.cat([pmtrx1.mT @ pmtrx2,
                          torch.repeat_interleave(cv1, n, 3),
                          torch.repeat_interleave(cv2.mT, m, 2)], 1)

    def step(self, batch, batch_n):  # TO DO
        pmtrx1, pmtrx2, amino1, amino2, cv_crop1, cv_crop2 = batch

        # if np.random.random() < amino_prob:
        code256_1 = self.model.encode_amino(amino1)
        code256_2 = self.model.encode_amino(amino2)
        # else:
        #     code256_1 = self.model.encode_cosma(pmtrx1.mT @ pmtrx1)
        #     code256_2 = self.model.encode_cosma(pmtrx2.mT @ pmtrx2)

        true = self.prepare_true_tensor(pmtrx1, pmtrx2, cv_crop1, cv_crop2)
        pred = self(amino1, code256_1, amino2, code256_2)

        return self.loss(pred, true)

    def forward(self, amino1, code256_1, amino2, code256_2):
        return self.model((amino1, code256_1, amino2, code256_2))


def train(checkpoint=None, full_cont=True):
    if checkpoint is None:
        pipeline = Full_Pipeline(train_path, valid_path)
    else:
        pipeline = Full_Pipeline.load_from_checkpoint(checkpoint,
                                                      train_pkl=train_path,
                                                      valid_pkl=valid_path)
    pipeline.set_crop(train_crop)
    print(f"Train set: {len(pipeline.train_set)}")
    print(f"Valid set: {len(pipeline.valid_set)}")
    callback = pl.callbacks.ModelCheckpoint(outdir_path, 'full_save',
                                            every_n_epochs=10)
    logger = pl.loggers.CSVLogger(outdir_path, 'logs',
                                  flush_logs_every_n_steps=5000)
    trainer = pl.Trainer(logger, callbacks=[callback],
                         accelerator=accel_hp, max_epochs=epochs)
    if checkpoint is None or not full_cont:
        trainer.fit(pipeline)
    else:
        trainer.fit(pipeline, ckpt_path=checkpoint)
    # trainer.save_checkpoint(save_path)
    return pipeline


if __name__ == '__main__':
    train('./full_log/full_save-100ep.ckpt')
