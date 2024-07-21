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
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from taming.data.utils import custom_collate

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module), cls)

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "-n", "--name", type=str, default="", nargs="?", 
        help="Postfix for logdir"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default="", nargs="?", 
        help="Resume from logdir or checkpoint in logdir"
    )
    parser.add_argument(
        "-b", "--base", nargs="*", metavar="base_config.yaml", 
        help="Paths to base configs, loaded left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=[]
    )
    parser.add_argument(
        "-t", "--train", type=str2bool, const=True, default=False, nargs="?", 
        help="Train"
    )
    parser.add_argument(
        "--no-test", type=str2bool, const=True, default=False, nargs="?", 
        help="Disable test"
    )
    parser.add_argument(
        "-p", "--project", help="Name of new or path to existing project"
    )
    parser.add_argument(
        "-d", "--debug", type=str2bool, nargs="?", const=True, default=False, 
        help="Enable post-mortem debugging"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=23, 
        help="Seed for seed_everything"
    )
    parser.add_argument(
        "-f", "--postfix", type=str, default="", 
        help="Post-postfix for default name"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, 
        help="Number of GPUs to use (default: 0)"
    )

    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--distributed_backend", type=str, default="ddp", help="Distributed backend")
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a PyTorch dataset."""
    def __init__(self, dataset):
        self.data = dataset

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
        if train:
            self.dataset_configs["train"] = train
        if validation:
            self.dataset_configs["validation"] = validation
        if test:
            self.dataset_configs["test"] = test
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = {k: instantiate_from_config(cfg) for k, cfg in self.dataset_configs.items()}
        if self.wrap:
            self.datasets = {k: WrappedDataset(v) for k, v in self.datasets.items()}

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate
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

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(
                self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml")
            )

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, f"{self.now}-lightning.yaml")
            )
        elif not self.resume and os.path.exists(self.logdir):
            dst = os.path.join(os.path.dirname(self.logdir), "child_runs", os.path.basename(self.logdir))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.rename(self.logdir, dst)
            except FileNotFoundError:
                pass

default_logger_cfgs = {
    "wandb": {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "project": "DALLE-Couture",
            "entity": "kairess",
            "name": None,  # Placeholder, set in the main block
            "save_dir": None,  # Placeholder, set in the main block
            "offline": None,  # Placeholder, set in the main block
            "id": None,  # Placeholder, set in the main block
        },
    },
    "tensorboard": {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "save_dir": None,  # Placeholder, set in the main block
            "name": "tensorboard",
        },
    },
}

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            WandbLogger: self._wandb,
            TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = {f"{split}/{k}": wandb.Image(torchvision.utils.make_grid(v)) for k, v in images.items()}
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k, v in images.items():
            grid = torchvision.utils.make_grid(v)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

def run_experiment(config, lightning_config, args, opt):
    seed_everything(config.seed, workers=True)
    pl.seed_everything(config.seed, workers=True)
    data_module = DataModuleFromConfig(**config.data)

    trainer_args = {
        "max_epochs": config.training.max_epochs,
        "log_every_n_steps": config.training.log_every_n_steps,
        "accumulate_grad_batches": config.training.accumulate_grad_batches,
        "callbacks": [],
        "precision": config.training.precision,
        "default_root_dir": opt.logdir,
        "resume_from_checkpoint": opt.resume,
        "gpus": config.training.gpus,
        "distributed_backend": config.training.distributed_backend,
    }

    if opt.resume:
        trainer_args["resume_from_checkpoint"] = opt.resume

    if nondefault_trainer_args(opt):
        print("Warning: Detected non-default trainer args")
        print(nondefault_trainer_args(opt))

    if config.callbacks:
        trainer_args["callbacks"] += [get_obj_from_str(cb["target"])(**cb.get("params", {})) for cb in config.callbacks]

    if config.loggers:
        loggers = [get_obj_from_str(logger["target"])(**logger.get("params", {})) for logger in config.loggers]
        trainer_args["logger"] = loggers[0] if len(loggers) == 1 else loggers

    trainer = Trainer(**trainer_args)

    if not config.callbacks:
        trainer.callbacks.append(
            SetupCallback(
                resume=opt.resume,
                now=opt.now,
                logdir=opt.logdir,
                ckptdir=opt.ckptdir,
                cfgdir=opt.cfgdir,
                config=config,
                lightning_config=lightning_config,
            )
        )

    if not opt.no_test and "test" in config.training and config.training.test.get("log_images"):
        trainer.callbacks.append(
            ImageLogger(
                batch_frequency=config.training.test.log_images.batch_frequency,
                max_images=config.training.test.log_images.max_images,
                clamp=config.training.test.log_images.clamp,
            )
        )

    if opt.train:
        trainer.fit(pl_module, datamodule=data_module)

    if not opt.no_test:
        trainer.test(pl_module, datamodule=data_module)

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, use -n/--name in combination with --resume_from_checkpoint."
        )
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = logdir.split("/")[logdir.split("/").index("logs") + 1]
    else:
        name = f"_{opt.name}" if opt.name else f"_{os.path.splitext(os.path.basename(opt.base[0]))[0]}" if opt.base else ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        model = instantiate_from_config(config.model)

        default_logger_cfg = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": "DALLE-Couture",
                    "entity": "kairess",
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                },
            }
        }
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg["wandb"], logger_cfg)
        trainer_kwargs = {"logger": instantiate_from_config(logger_cfg)}

        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {"logging_interval": "step"},
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()

        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(",")) if not cpu else 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        assert accumulate_grad_batches > 0, "Please set accumulate_grad_batches in the config file or command-line arguments."
        if "distributed_backend" in trainer_opt:
            accumulate_grad_batches *= ngpu
        model.configure_optimizers(
            base_lr=base_lr, bs=bs, accumulate_grad_batches=accumulate_grad_batches
        )

        trainer.fit(model, data)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
