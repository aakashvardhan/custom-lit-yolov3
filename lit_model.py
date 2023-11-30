import torch
import lightning as pl
import utils
from tqdm import tqdm
from models.yolo import YOLOv3
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import loss

class YOLOv3LightningModule(pl.LightningModule):
    def __init__(self, config, lr = 1e-4, one_cycle_best_LR = 1e-4):
        super().__init__()
        self.config = config
        self.lr = lr
        self.one_cycle_best_LR = one_cycle_best_LR
        self.model = YOLOv3(num_classes = config.NUM_CLASSES)
        self.automatic_optimization = True
        self.loss_fn = loss.YoloLoss()
        
    def forward(self, x):
        return self.model(x)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.config.WEIGHT_DECAY)
        EPOCHS = self.config.NUM_EPOCHS * 2 // 5
        scheduler = OneCycleLR(optimizer, max_lr = self.one_cycle_best_LR, epochs =EPOCHS , steps_per_epoch = len(self.trainer.datamodule.train_dataloader()),
                               pct_start = 5/EPOCHS, anneal_strategy = 'linear')
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        y0, y1, y2 = (targets[0].to(self.device),targets[1].to(self.device),targets[2].to(self.device))
        out = self(images)

        loss = (self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2]))
        
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        y0, y1, y2 = (targets[0].to(self.device),
                      targets[1].to(self.device),
                      targets[2].to(self.device))
        out = self(images)

        loss = (self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2]))
        
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
    def on_train_start(self):
        if self.config.LOAD_MODEL:
            utils.load_checkpoint(
                self.config.CHECKPOINT_FILE, self.model, self.optimizers(), self.config.LEARNING_RATE
            )

        self.scaled_anchors = (
            torch.tensor(self.config.ANCHORS)
            * torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)