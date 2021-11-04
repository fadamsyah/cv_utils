import pytorch_lightning as pl

from torch.utils.data import DataLoader

class CommonDataModuleWrapper(pl.LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None,
                 test_dataset=None):
        super().__init__()
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.loader_setting = {
            'train': {
                'batch_size':8, 'shuffle':True,
                'num_workers':4, 'drop_last':True
            },
            'val': {
                'batch_size':32, 'shuffle':False,
                'num_workers':4, 'drop_last':False
            },
            'test': {
                'batch_size':32, 'shuffle':False,
                'num_workers':4, 'drop_last':False
            }
        }
    
    def configure_loader_setting(self, setting):
        for set_name, params in setting.items():
            for p_name, p_val in params.items():
                self.loader_setting[set_name][p_name] = p_val
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_setting['train'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader_setting['val'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader_setting['test'])