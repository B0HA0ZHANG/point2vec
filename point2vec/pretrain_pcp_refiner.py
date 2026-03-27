from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from point2vec.datasets import (
    ModelNet40Ply2048DataModule,
    ScanObjectNNDataModule,
    ShapeNet55DataModule,
)
from point2vec.models.point2vec_pcp_refiner import Point2VecPCPRefiner

if __name__ == "__main__":
    cli = LightningCLI(
        Point2VecPCPRefiner,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 800,
            "track_grad_norm": 2,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 200,
            "callbacks": [
                LearningRateMonitor(),
                ModelCheckpoint(save_on_train_epoch_end=True),
            ],
        },
        seed_everything_default=0,
        save_config_callback=None,
    )
