import argparse
import os
import datetime
import glob
import importlib
import numpy as np
from PIL import Image

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf

def custom_collate(batch):
    return batch

class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(
                self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now))
            )

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torch.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.log(grids)

    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torch.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0
            grid = grid.numpy()
            grid = np.moveaxis(grid, 0, -1)
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if hasattr(pl_module, "log_images") and callable(pl_module.log_images) and (
            batch_idx % self.batch_freq == 0 and batch_idx > 0
            or batch_idx in self.log_steps
            or (split == "val" and batch_idx == 0)
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)
            self.logger_log_images[logger](pl_module, images, batch_idx, split)
            self.log_local(
                pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
            )
            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    pass

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
        type=str,
        default="",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = get_parser()
    args = parser.parse_args([])
    return sorted(
        k for k in vars(args) if getattr(opt, k, None) != getattr(args, k, None)
    )

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError("Cannot specify both --name and --resume.")
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find the requested resume checkpoint")
        logdir = opt.resume.rstrip("/")
        opt.name = logdir.split("/")[-1]
    else:
        opt.name = now + opt.postfix
        logdir = os.path.join(opt.project, opt.name)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    seed_everything(opt.seed)

    try:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
    except Exception:
        configs = list()
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = pl.Trainer.from_argparse_args(trainer_opt)
    def nondefault_trainer_args(opt):
        parser = get_parser()
        args = parser.parse_args([])
    
    # Ensure both opt and args have the same set of attributes
        opt_vars = vars(opt)
        args_vars = vars(args)

    # Set default None for any missing attributes
        for key in set(opt_vars.keys()).union(args_vars.keys()):
            if key not in opt_vars:
                opt_vars[key] = None
            if key not in args_vars:
                args_vars[key] = None

        return sorted(
            k for k in args_vars if opt_vars[k] != args_vars[k]
        )


    # Remove 'gpus' key from trainer_config if it exists
    if "gpus" in trainer_config:
        del trainer_config["gpus"]

    # Update trainer_config with non-default arguments from opt
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    # Set default_root_dir in trainer_config if not already set
    if "default_root_dir" not in trainer_config:
        trainer_config["default_root_dir"] = None  # Set your default root directory here

    # Set up checkpoint and logger configurations
    trainer_kwargs = {}

    # Set up logger
    if "logger" in lightning_config:
        trainer_kwargs["logger"] = instantiate_from_config(lightning_config.logger)

    # Default ModelCheckpoint callback
    default_checkpoint_callback = ModelCheckpoint(
        dirpath=None,  # Set your checkpoint directory here
        filename="{epoch}-{val_loss:.2f}",  # Set your checkpoint filename format here
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
    )

    if "modelcheckpoint" in lightning_config:
        default_checkpoint_callback = instantiate_from_config(lightning_config.modelcheckpoint)

    # Set up callbacks
    trainer_kwargs["callbacks"] = [
        SetupCallback(opt.resume, now, logdir=None, ckptdir=None, cfgdir=None, config=config, lightning_config=lightning_config),
        default_checkpoint_callback
    ]

    if "callbacks" in lightning_config:
        trainer_kwargs["callbacks"] += [
            instantiate_from_config(callback_conf)
            for callback_conf in lightning_config.callbacks.values()
        ]

    # Initialize DataModule
    data = instantiate_from_config(config.data)

    # Initialize Model
    model = instantiate_from_config(config.model)

    # Initialize Trainer
    trainer = Trainer(
        logger=trainer_kwargs.get("logger", None),
        callbacks=trainer_kwargs["callbacks"],
        **trainer_config
    )

    # Run training or testing based on command-line arguments
    if opt.train:
        trainer.fit(model, datamodule=data)
    elif not opt.no_test:
        trainer.test(model, datamodule=data)