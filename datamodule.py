from typing import Optional

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from dataset import KineticsDataset

class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, conf: dict):
        super().__init__()
        self.modality = conf["modality"]
        self.train_bs = conf["optimization"]["train_bs"]
        self.val_bs = conf["optimization"]["val_bs"]
        self.conf = conf

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = KineticsDataset("train", self.train_bs)
        self.val_dataset = KineticsDataset("val", self.val_bs)

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=False,
            num_workers=self.conf["training"]["train_num_workers"],
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=self.conf["training"]["val_num_workers"],
            pin_memory=True,
            drop_last=True
    )