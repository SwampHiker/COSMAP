#**********************************************#
# The file with some low quality research code #
#   on which other scripts rely.               #
#**********************************************#

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cosma_algebra as ca
import rectifier as rct
import pickle as pkl
import numpy as np
import torch
import time

# For FG_Pipeline...
batch_hp = 128
workers_hp = 8
learning_rate_hp = 1e-3
device_hp = 'cuda'
accel_hp = 'gpu'
train_path = './train.pkl'
valid_path = './valid.pkl'
outdir_path = './fg_log/'
epochs = 1
# save_path = './fg_save.pkl'
train_crop = 64

class FG_Dataset(Dataset):
    """Base dataset."""
    
    def __init__(self, file_pkl):
        super().__init__()
        with open(file_pkl, 'rb') as f:
            self.raw_data = pkl.load(f)
        self.set_crop(verbose=False)

    def set_crop(self, crop=None, verbose=True):
        if verbose:
            print('Disabling crop...' if crop is None else f"Setting crop {crop}...")
        self.crop = crop
        if crop is None:
            self.data = self.raw_data
        else:
            self.data = [ch for ch in self.raw_data if ch['crop'] >= crop + 1]
        # if verbose:
        #     print('Crop is set.')

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def load_item(self, item):
        return self.data[item]

    def __getitem__(self, item):
        # return self.data[item] #TO CHANGE
        chain_dict = self.load_item(item)
        if self.crop is None:
            pmtrx_crop = chain_dict['pmtrx']
        else:
            k = np.random.choice(range(chain_dict['crop'] - self.crop))
            pmtrx_crop = chain_dict['pmtrx'][:, k:k+self.crop]
        # amino_crop = chain_dict['amino'][k:k+self.crop+1]
        # return {'cosma': np.expand_dims(ca.cosma_from_pmtrx(pmtrx_crop, pmtrx_crop).astype(np.float32), 0),
        #         'amino': ca.amino_list_to_indices_stack(amino_crop, True).astype(np.int64)}
        return np.expand_dims(ca.cosma_from_pmtrx(pmtrx_crop, pmtrx_crop).astype(np.float32), 0)

class FG_Pipeline(pl.LightningModule):
    """Base pipeline"""

    def __init__(self, train_pkl = None, valid_pkl = None): #maybe smth to add
        super().__init__()
        self.train_set = None if train_pkl is None else FG_Dataset(train_pkl)
        self.valid_set = None if valid_pkl is None else FG_Dataset(valid_pkl)
        self.model = None # rct.CosMaP_AutoEncode() #rct.CosMaP_Design()
        if not self.train_set is None:
            self.train_load = DataLoader(self.train_set, batch_hp, num_workers=workers_hp, pin_memory=False)
        if not self.valid_set is None:
            self.valid_load = DataLoader(self.valid_set, batch_hp, num_workers=workers_hp, pin_memory=False)
        self.loss = rct.SSIM_loss(1) #nn.CrossEntropyLoss() #nn.NLLLoss()
        self.timer = time.time()
        self.curr_epoch = 0

    def forward(self, cosma):
        return self.model(cosma)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), learning_rate_hp)

    def step(self, batch, batch_nb):
        # cosma, amino = batch.values()
        # pred = self.model(cosma).mT
        # b, c, m = pred.shape
        # loss = self.loss(torch.cat([torch.zeros((b, 1, m), device=device_hp), pred[:, 0:20, :]], 1), amino[:, 0, :]
        #     ) + self.loss(torch.cat([torch.zeros((b, 1, m), device=device_hp), pred[:, 20:, :]], 1), amino[:, 1, :])
        # return loss / 2
        return self.loss(self.model(batch), batch)


    def training_step(self, batch, batch_nb):
        loss = self.step(batch, batch_nb)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.step(batch, batch_nb)
        self.log('validation_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # def on_training_epoch_end(self, outputs):
    #     print(f'\n\ttraining {self.curr_epoch} end\t\ttime = {(time.time() - self.timer) / 60} m')
    #     self.curr_epoch += 1
    #     self.timer = time.time()

    def train_dataloader(self):
        return self.train_load

    def val_dataloader(self):
        return self.valid_load

    def set_crop(self, crop = None):
        self.train_set.set_crop(crop)
        self.valid_set.set_crop(crop)

class SG_Dataset(FG_Dataset):
    """Dataset for fold prediction."""
    
    def __init__(self, file_pkl):
        super().__init__(file_pkl)

    def __getitem__(self, item):
        # return self.data[item] #TO CHANGE
        chain_dict = self.load_item(item)
        if self.crop is None:
            pmtrx_crop = chain_dict['pmtrx']
            amino_crop = chain_dict['amino']
        else:
            k = np.random.choice(range(chain_dict['crop'] - self.crop))
            pmtrx_crop = chain_dict['pmtrx'][:, k:k+self.crop].astype(np.float32)                           
            amino_crop = chain_dict['amino'][k:k+self.crop+1]
        return {'cosma': np.expand_dims(ca.cosma_from_pmtrx(pmtrx_crop, pmtrx_crop), 0), 
                'amino': ca.amino_list_to_array_stack(amino_crop, False).astype(np.float32)}

class Gen3_Dataset(Dataset):
    """Dataset for interaction prediction."""
    
    def __init__(self, file_pkl):
        super().__init__()
        with open(file_pkl, 'rb') as f:
            self.raw_data = pkl.load(f)
        self.set_crop(verbose=False)

    def set_crop(self, crop=None, verbose=True):
        if verbose:
            print('Disabling crop...' if crop is None else f"Setting crop {crop}...")
        if crop is None:
            self.crop = None
            self.data = self.raw_data
        else:
            self.crop = crop if type(crop) is tuple else (crop, crop)
            self.data = [ch for ch in self.raw_data if ch['crop'][0] >= self.crop[0] + 1 and ch['crop'][1] >= self.crop[1] + 1]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def load_item(self, item):
        return self.data[item]

    def __getitem__(self, item, shuffle=True):
        chain_dict = self.load_item(item)
        pmtrx_crop1, pmtrx_crop2 = chain_dict['pmtrx']
        amino_crop1, amino_crop2 = chain_dict['amino']

        if not self.crop is None:
            i = np.random.choice(range(chain_dict['crop'][0] - self.crop[0]))
            j = np.random.choice(range(chain_dict['crop'][1] - self.crop[1]))            
            pmtrx_crop1 = pmtrx_crop1[:, i:i+self.crop[0]]
            pmtrx_crop2 = pmtrx_crop2[:, j:j+self.crop[1]]
            amino_crop1 = amino_crop1[i:i+self.crop[0]+1]
            amino_crop2 = amino_crop2[j:j+self.crop[1]+1]

        pmtrx_crop1 = np.expand_dims(pmtrx_crop1.astype(np.float32), 0)
        pmtrx_crop2 = np.expand_dims(pmtrx_crop2.astype(np.float32), 0)

        amino_crop1 = ca.amino_list_to_array_stack(amino_crop1, False).astype(np.float32)
        amino_crop2 = ca.amino_list_to_array_stack(amino_crop2, False).astype(np.float32)        

        # return {'pmtrx1': pmtrx_crop1, 'pmtrx2': pmtrx_crop2,
        #         'amino1': amino_crop1, 'amino2': amino_crop2}

        return (pmtrx_crop2, pmtrx_crop1, amino_crop2, amino_crop1) if shuffle and np.random.rand() < 0.5 else (pmtrx_crop1, pmtrx_crop2, amino_crop1, amino_crop2)
