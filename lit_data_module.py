# Import LightningDataModule from pytorch_lightning
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from voc_coco_dataset import YOLODataset


class YOLODataModule(LightningDataModule):
    def __init__(self, )

