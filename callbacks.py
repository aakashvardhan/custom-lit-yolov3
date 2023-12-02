import torch
import lightning as pl
from utils import (
    check_class_accuracy,
    mean_average_precision,
    plot_couple_examples,
    get_evaluation_bboxes,
)
from tqdm.notebook import tqdm


class ClassAccuracyCallback(pl.Callback):
    def __init__(self, config, train_n_epochs: int = 1, test_n_epochs: int = 5):
        super().__init__()
        self.train_n_epochs = train_n_epochs
        self.test_n_epochs = test_n_epochs
        self.config = config
        
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.train_n_epochs == 0:
            print("Calculating training accuracy...")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model=pl_module,
                                 loader= trainer.datamodule.train_dataloader(),
                                 threshold=self.config.CONF_THRESHOLD)
            pl_module.log_dict(
                {
                    "train_class_accuracy": class_acc,
                    "train_no_object_accuracy": no_obj_acc,
                    "train_object_accuracy": obj_acc,
                },
                logger=True,
            )
            
            print(f"Train accuracy: {class_acc.item():.3f}")
            print(f"No obj accuracy: {no_obj_acc.item():.3f}")
            print(f"Obj accuracy: {obj_acc.item():.3f}")
        
        if (trainer.current_epoch + 1) % self.test_n_epochs == 0:
            print("Calculating Test accuracy...")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model=pl_module,
                                 loader= trainer.datamodule.test_dataloader(),
                                 threshold=self.config.CONF_THRESHOLD)
            
            pl_module.log_dict(
                {
                    "test_class_accuracy": class_acc,
                    "test_no_object_accuracy": no_obj_acc,
                    "test_object_accuracy": obj_acc,
                },
                logger=True,
            )
            
            print(f"Test accuracy: {class_acc.item():.3f}")
            print(f"No obj accuracy: {no_obj_acc.item():.3f}")
            print(f"Obj accuracy: {obj_acc.item():.3f}")
            
            
class MAPCallback(pl.Callback):
    def __init__(self, config,test_n_epochs: int = 5):
        super().__init__()
        self.test_n_epochs = test_n_epochs
        self.config = config
        
    def on_train_epoch_end(self, trainer, pl_module): 
        if (trainer.current_epoch + 1) % self.test_n_epochs == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                loader=trainer.datamodule.test_dataloader(),
                model=pl_module,
                iou_threshold=self.config.NMS_IOU_THRESH,
                anchors=self.config.ANCHORS,
                threshold=self.config.CONF_THRESHOLD,
                device=self.config.DEVICE,
            )

            map_val = mean_average_precision(
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                iou_threshold=self.config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=self.config.NUM_CLASSES,)
            
            
            pl_module.log("val_map", map_val.item(), logger=True)
            print(f"MAP: {map_val.item()}")
            pl_module.train()
            
class PlotExampleCallback(pl.Callback):
    def __init__(self, config, n_epochs: int = 5):
        super().__init__()
        self.n_epochs = n_epochs
        self.config = config
        
    def on_train_epoch_end(self, trainer, pl_module): 
        if (trainer.current_epoch + 1) % self.n_epochs == 0:
            print("Plotting example...")
            plot_couple_examples(
                pl_module,
                loader=trainer.datamodule.test_dataloader(),
                iou_thresh=0.5,
                anchors=pl_module.scaled_anchors,
                thresh=0.6
            )