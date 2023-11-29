# Import LightningDataModule from pytorch_lightning
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from voc_coco_dataset import YOLODataset


class YOLODataModule(LightningDataModule):
    def __init__(self, config, train_dir, test_dir):
        super().__init__()
        self.config = config
        self.train_dir = train_dir
        self.test_dir = test_dir
        
    def setup(self, stage = 'None'):
        self.voc_train = YOLODataset(
            self.train_dir,
            transform=self.config.train_transforms,
            S = [
                self.config.IMAGE_SIZE // 32,
                self.config.IMAGE_SIZE // 16,
                self.config.IMAGE_SIZE // 8,
            ],
            img_dir=self.config.IMG_DIR,
            label_dir=self.config.LABEL_DIR,
            anchors=self.config.ANCHORS,
            _mosaic_prob=self.config.MOSAIC_PROB
        )
        
        self.voc_test = YOLODataset(
            self.test_dir,
            transform=self.config.test_transforms,
            S = [
                self.config.IMAGE_SIZE // 32,
                self.config.IMAGE_SIZE // 16,
                self.config.IMAGE_SIZE // 8,
            ],
            img_dir=self.config.IMG_DIR,
            label_dir=self.config.LABEL_DIR,
            anchors=self.config.ANCHORS,
            _mosaic_prob=0
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
        
        

