import os

import lightning.pytorch as pl

import ray.train
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

def train_func():
    # [1] Create a Lightning model
    model = MyLightningModule(lr=1e-3, feature_dim=128)

    # [2] Report Checkpoint with callback
    ckpt_report_callback = RayTrainReportCallback()

    # [3] Create a Lighting Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=100,
        # New configurations below
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[ckpt_report_callback],
    )

    # Validate your Lightning trainer configuration
    trainer = prepare_trainer(trainer)

    # [4] Build your datasets on each worker
    datamodule = MyLightningDataModule(batch_size=32)
    trainer.fit(model, datamodule=datamodule)

# [5] Explicitly define and run the training function
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ray.train.ScalingConfig(num_workers=4, use_gpu=True),
    run_config=ray.train.RunConfig(
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_accuracy",
            checkpoint_score_order="max",
        ),
    )
)
result = ray_trainer.fit()

# [6] Load the trained model from a simplified checkpoint interface.
checkpoint: ray.train.Checkpoint = result.checkpoint
with checkpoint.as_directory() as checkpoint_dir:
    print("Checkpoint contents:", os.listdir(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
    model = MyLightningModule.load_from_checkpoint(checkpoint_path)