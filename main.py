import argparse
import os
import sys
import datetime
import glob
import importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from taming.data.utils import custom_collate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
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
    parser.add_argument(
        "--strategy",
        type=str,
        help="Distributed training strategy (e.g., ddp, dp)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        help="Comma-separated list of GPU devices (e.g., 0,1,2,3)",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # Manually define Trainer arguments
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--distributed_backend", type=str, default=None)
    # Add other Trainer arguments here
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
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
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


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

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
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
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            pl_module.logger.experiment.add_image(f"{split}/{k}", grid, pl_module.global_step)

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.batch_freq != 0:
            return
        images = pl_module.log_images(trainer)
        if images:
            split = "train" if pl_module.training else "val"
            for logger in trainer.logger:
                if type(logger) in self.logger_log_images:
                    self.logger_log_images[type(logger)](pl_module, images, trainer.global_step, split)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # Manually define Trainer arguments
    parser.add_argument("--devices", type=int, nargs='+', default=None, help="List of GPUs to use")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None, help="Distributed training strategy")
    # Add other Trainer arguments here
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load config
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    if args.config:
        extra_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, extra_config)
    config = OmegaConf.create(config)
    if hasattr(config, "seed"):
        seed_everything(config.seed)

    # Initialize data module
    data_module = DataModuleFromConfig(
        batch_size=config.get("batch_size", 32),
        train=config.get("train"),
        validation=config.get("validation"),
        test=config.get("test"),
        wrap=config.get("wrap", False),
        num_workers=config.get("num_workers", None)
    )

    # Initialize trainer
    trainer_args = {k: getattr(args, k) for k in nondefault_trainer_args(args)}
    trainer = pl.Trainer(**trainer_args)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=os.path.join(args.logdir, "checkpoints")
    )

    # Define learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Define additional callbacks
    setup_callback = SetupCallback(
        resume=args.resume,
        now=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        logdir=args.logdir,
        ckptdir=os.path.join(args.logdir, "checkpoints"),
        cfgdir=os.path.join(args.logdir, "config"),
        config=config,
        lightning_config=OmegaConf.create(vars(args))
    )
    
    image_logger = ImageLogger(
        batch_frequency=500,
        max_images=16
    )

    trainer.callbacks = [checkpoint_callback, lr_monitor, setup_callback, image_logger]

    if args.train:
        model = pl.LightningModule()  # Replace with actual model class
        trainer.fit(model, datamodule=data_module)

    if not args.no_test:
        model = pl.LightningModule()  # Replace with actual model class
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    trainer_args = {
        'accelerator': 'gpu',
        'devices': list(map(int, args.devices.split(','))) if args.devices else None,
        'strategy': args.strategy,
        'max_epochs': args.max_epochs,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'distributed_backend': args.distributed_backend,
       
    }
    trainer = pl.Trainer(**trainer_args)