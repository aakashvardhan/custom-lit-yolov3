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
        self.voc_train, _, _ = utils.get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )
        
        
        _, self.voc_test, _ = utils.get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.voc_train,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.voc_test,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
        
        

