from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch

def train_model(model_class, model_name, train_dataloader, val_dataloader, seed=42, epochs=20, logs_path='logs', hyperparameters={}):

    seed_everything(seed, workers=True)

    model = model_class(**hyperparameters)

    logger = TensorBoardLogger(logs_path, name=f"{model_name}_seed{seed}")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=None,
        filename=f'{model_name}-seed={seed}' + '-{epoch:02d}-{val_loss:.2f}-{val_f1_score:.2f}',
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True
    )

    trainer.fit(model, train_dataloader, val_dataloader)