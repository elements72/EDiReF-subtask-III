import json
import os
from collections import OrderedDict

import lightning as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from typing import Generic, TypeVar

def test_model(model, test_loader):
    trainer = pl.Trainer()
    trainer.test(model, test_loader)

def train_model(model_class, model_name, train_loader, val_loader, seed=42, epochs=20, logs_path='logs',
                hyperparameters=None, wandb=None) -> pl.LightningModule:
    if hyperparameters is None:
        hyperparameters = {}

    seed_everything(seed, workers=True)

    if os.path.exists("hyperparams.json"):
        print("Loading hyperparameters from file...")
        with open('hyperparams.json', 'r') as f:
            file = json.load(f)
            try:
                hyperparameters[model_name]["lr"] = file[model_name]["lr"]
            except KeyError:
                print(f"Hyperparameters for model {model_name} not found. Using default values.")
    else:
        print("No hyperparameters file found. Using default values.")

    model = model_class(**hyperparameters)

    # Create wandb logger
    wandb_logger = WandbLogger

    wandb_logger = WandbLogger(log_model="all", project="EDiReF-subtask-III", name=f'{model_name}-seed={seed}', )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=logs_path,
        filename=f'{model_name}-seed={seed}' + '-{epoch:02d}-{val_loss:.2f}',
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
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=False
    )

    trainer.fit(model, train_loader, val_loader)
    wandb_logger.experiment.finish()
    return model

def train_model_seeds(model_class, model_name, train_loader, val_loader, seeds, epochs=20, logs_path='logs', hyperparameters=None, wandb=None):
    # Set seeds
    for seed in seeds:
        print("#" * 50)
        print(f"Training model {model_name} with seed {seed}")
        print("#" * 50)
        train_model(model_class, model_name, train_loader, val_loader, seed, epochs, logs_path, hyperparameters, wandb)
    return 


def hyperparameters_tuning(model_class, model_name, datamodule, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {}

    # Check if the hyperparameters of the model are already in the file
    if os.path.exists("hyperparams.json"):
        with open('hyperparams.json', 'r') as f:
            file = json.load(f)
            if file.get(model_name) is not None:
                print(f"Hyperparameters for model {model_name} already found. Skipping...")
                return

    PERCENT_VALID_EXAMPLES = 0.1  # increase if you want to include more validation samples
    EPOCHS = 5

    optim_lr_rate = {}

    model = model_class(**hyperparameters)
    trainer = Trainer(
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        accelerator="auto",
        max_epochs=EPOCHS,
    )

    # Create the Tuner
    tuner = pl.pytorch.tuner.Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule)

    model.hparams.lr = lr_finder.suggestion()
    print("#########################")
    print(f'Auto-find model LR is:\n [Model Name: {model_name}, Best Lr: {model.hparams.lr} ]')
    print("#############################")

    # append in dict
    optim_lr_rate[model_name] = {}
    optim_lr_rate[model_name]['lr'] = model.hparams.lr
    # fig = lr_finder.plot(suggest=True)

    print(optim_lr_rate)
    if os.path.exists("hyperparams.json"):
        with open('hyperparams.json', 'r+') as f:
            file = json.load(f)
            file.update(optim_lr_rate)
            f.seek(0)
            json.dump(file, f, indent=2)
    else:
        with open('hyperparams.json', 'w') as f:
            json.dump(optim_lr_rate, f, indent=2)



K = TypeVar('K')
V = TypeVar('V')
class FIFOCache(Generic[K, V]):
    """
    A FIFO cache that stores the last n elements.
    The keys are of type K and the values are of type V.
    """

    def __init__(self, size: int):
        self.size = size
        self.cache = OrderedDict()

    def contains(self, key: K) -> bool:
        """
        Checks if the cache contains the given key.
        """

        return key in self.cache

    def get(self, key: K) -> V:
        """
        Returns the value associated with the given key.
        If the key is not in the cache, None is returned.
        """

        return self.cache.get(key, None)

    def put(self, key: K, value: V) -> None:
        """
        Inserts the given key-value pair into the cache.
        If the key is already in the cache, its value is updated.
        If the cache is full, the oldest key-value pair is removed.
        """

        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        if len(self.cache) > self.size:
            self.cache.popitem(last=False)

    def __repr__(self) -> str:
        s = "Cache:(\n"
        for key, value in self.cache.items():
            s += f"\t'{key}': {value.shape}\n"

        s += ")"
        return s
