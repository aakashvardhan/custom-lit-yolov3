# Import LightningDataModule from pytorch_lightning
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from voc_coco_dataset import YOLODataset
import utils

class YOLODataModule(LightningDataModule):
    def __init__(self, config, train_dir, test_dir):
        super().__init__()
        self.config = config
        self.train_dir = train_dir
        self.test_dir = test_dir
        
    def setup(self, stage = 'None'):
        self.voc_train, self.voc_test, self.voc_val = utils.get_loaders(
            train_csv_path=self.train_dir,
            test_csv_path=self.test_dir,
        )
        
    def train_dataloader(self):
        return self.voc_train
        
    def test_dataloader(self):
        return self.voc_test
    
    def val_dataloader(self):
        return self.voc_val
        
        

