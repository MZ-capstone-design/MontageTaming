import argparse, os, sys, datetime, glob, importlib
# argparse: 커맨드 라인 인자 파싱을 위한 라이브러리
# os: 운영체제와 상호작용하기 위한 라이브러리 (파일 및 디렉토리 조작)
# sys: 파이썬 인터프리터와 상호작용하기 위한 라이브러리 (명령줄 인자, 경로 등)
# datetime: 날짜와 시간을 다루기 위한 라이브러리
# glob: 파일 패턴 매칭을 위한 라이브러리 (예: 특정 확장자를 가진 모든 파일 찾기)
# importlib: 동적으로 모듈을 임포트하고, 재로드하기 위한 라이브러리

from omegaconf import OmegaConf
# OmegaConf: 구성 파일(.yaml 등)을 다루기 위한 라이브러리

import numpy as np
# numpy: 수치 연산을 위한 라이브러리 (특히 배열 및 행렬 연산)

from PIL import Image
# PIL (Python Imaging Library): 이미지 파일을 열고, 조작하기 위한 라이브러리

import torch
# PyTorch: 딥러닝을 위한 메인 프레임워크

import torchvision
# torchvision: PyTorch에서 컴퓨터 비전 작업을 위한 유틸리티 (모델, 데이터셋, 전처리 등)

from torch.utils.data import random_split, DataLoader, Dataset
# random_split: 데이터셋을 랜덤하게 분할하는 유틸리티
# DataLoader: 데이터셋을 배치 단위로 로드하기 위한 유틸리티
# Dataset: PyTorch 데이터셋의 기본 클래스

import pytorch_lightning as pl
# PyTorch Lightning: PyTorch의 경량화된 상위 추상화 프레임워크

from pytorch_lightning import seed_everything
# seed_everything: 재현성을 위한 시드 설정 유틸리티

from pytorch_lightning.trainer import Trainer
# Trainer: PyTorch Lightning에서 훈련 과정을 관리하는 클래스

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# ModelCheckpoint: 모델 체크포인트를 저장하기 위한 콜백
# Callback: 훈련 과정에서 특정 이벤트 발생 시 호출되는 콜백의 기본 클래스
# LearningRateMonitor: 학습률 모니터링을 위한 콜백

from pytorch_lightning.utilities.distributed import rank_zero_only
# rank_zero_only: 분산 학습에서 랭크 0 프로세스에서만 실행되는 코드 유틸리티

import wandb
# wandb: 모델 훈련 및 실험 추적을 위한 툴 (Weights & Biases)

from taming.data.utils import custom_collate
# custom_collate: 데이터 로딩 시 사용될 커스텀 콜레이트 함수 (특정 데이터셋 전처리에 사용될 가능성 있음)



def get_obj_from_str(string, reload=False):
    # 문자열로부터 모듈과 클래스를 동적으로 가져오는 함수
    # reload가 True인 경우, 모듈을 다시 로드하여 최신 상태로 반영

    module, cls = string.rsplit(".", 1)
    # 문자열을 마지막 점(.)을 기준으로 모듈 이름과 클래스 이름으로 분리
    # 예: 'package.module.ClassName' -> module: 'package.module', cls: 'ClassName'

    if reload:
        module_imp = importlib.import_module(module)
        # 모듈을 동적으로 임포트

        importlib.reload(module_imp)
        # 모듈을 다시 로드하여 최신 상태로 갱신

    return getattr(importlib.import_module(module, package=None), cls)
    # 모듈을 임포트하고, 해당 모듈에서 클래스 객체를 동적으로 가져와 반환
    # 'getattr'을 사용하여 클래스 이름으로 클래스 객체를 반환



def get_parser(**parser_kwargs):
    # 커맨드 라인 인자를 파싱하기 위한 ArgumentParser 인스턴스를 생성하는 함수

    def str2bool(v):
        # 문자열을 boolean 값으로 변환하는 함수
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        # 유효한 boolean 문자열이 아닌 경우 예외 발생

    parser = argparse.ArgumentParser(**parser_kwargs)
    # ArgumentParser 인스턴스를 생성하고, 추가적인 파서 인자를 전달

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    # -n 또는 --name: 로그 디렉토리에 추가할 접미사 (기본값은 빈 문자열)

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    # -r 또는 --resume: 로그 디렉토리 또는 체크포인트에서 재개할 때 사용 (기본값은 빈 문자열)

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    # -b 또는 --base: 기본 구성 파일 경로들 (왼쪽에서 오른쪽으로 로드됨)
    #                명령줄 옵션으로 덮어쓰거나 추가 가능

    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    # -t 또는 --train: 훈련 모드 여부를 boolean 값으로 지정 (기본값은 False)

    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    # --no-test: 테스트를 비활성화할지 여부를 boolean 값으로 지정 (기본값은 False)

    parser.add_argument(
        "-p", 
        "--project", 
        help="name of new or path to existing project"
    )
    # -p 또는 --project: 새로운 프로젝트 이름 또는 기존 프로젝트 경로 지정

    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    # -d 또는 --debug: 디버깅 모드 활성화 여부를 boolean 값으로 지정 (기본값은 False)

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    # -s 또는 --seed: seed_everything에 사용할 시드 값 지정 (기본값은 23)

    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    # -f 또는 --postfix: 기본 이름에 추가할 후속 접미사 (기본값은 빈 문자열)

    return parser
    # ArgumentParser 인스턴스를 반환



def nondefault_trainer_args(opt):
    # 주어진 옵션(opt)과 PyTorch Lightning Trainer의 기본 인자를 비교하여
    # 기본값과 다른 인자들을 반환하는 함수

    parser = argparse.ArgumentParser()
    # ArgumentParser 인스턴스를 생성

    parser = Trainer.add_argparse_args(parser)
    # PyTorch Lightning의 Trainer 클래스에서 기본 인자들을 추가하여
    # ArgumentParser에 포함시킴
    # `add_argparse_args` 메서드는 Trainer의 기본 인자들을 추가함

    args = parser.parse_args([])
    # 빈 리스트를 인자로 전달하여 기본값만을 가진 argparse.Namespace 객체를 생성
    # `args`는 Trainer의 기본 인자들의 기본값을 가진 객체

    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
    # opt 객체와 `args` 객체의 각 인자 값을 비교
    # 기본값과 다른 인자들의 이름을 정렬된 리스트로 반환
    # `vars(args)`는 `args` 객체의 속성을 딕셔너리 형태로 반환
    # `getattr(opt, k)`는 `opt` 객체의 속성 값을 반환
    # `getattr(args, k)`는 `args` 객체의 속성 값을 반환



def instantiate_from_config(config):
    # 주어진 설정(config)으로부터 객체를 인스턴스화하는 함수

    if not "target" in config:
        # 설정(config)에 "target" 키가 없으면
        raise KeyError("Expected key `target` to instantiate.")
        # "target" 키가 필수라는 예외를 발생시킴

    # "target" 키에 해당하는 문자열로부터 객체를 가져와 인스턴스화
    # "params" 키가 있으면 해당 매개변수로 인스턴스를 생성
    # "params" 키가 없으면 빈 딕셔너리로 인스턴스를 생성
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



class WrappedDataset(Dataset):
    """__len__ 및 __getitem__ 메서드를 가진 임의의 객체를 PyTorch 데이터셋으로 감싸는 클래스"""

    def __init__(self, dataset):
        # 초기화 메서드: 데이터셋 객체를 클래스에 저장
        self.data = dataset
        # 주어진 데이터셋 객체를 `self.data`에 저장

    def __len__(self):
        # 데이터셋의 길이를 반환하는 메서드
        return len(self.data)
        # `self.data`의 길이를 반환 (데이터셋의 총 항목 수)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 항목을 반환하는 메서드
        return self.data[idx]
        # `self.data`에서 주어진 인덱스의 항목을 반환
        # 인덱싱을 통해 데이터를 반환 (예: 데이터셋에서 특정 항목을 가져옴)



class DataModuleFromConfig(pl.LightningDataModule):
    """구성(config)에서 데이터셋과 데이터 로더를 설정하는 PyTorch Lightning 데이터 모듈 클래스"""

    def __init__(
        self, batch_size, train=None, validation=None, test=None, wrap=False, num_workers=None
    ):
        # 초기화 메서드
        # 배치 사이즈, 데이터셋 구성, 데이터 로더의 워커 수 등을 설정
        super().__init__()
        self.batch_size = batch_size
        # 배치 사이즈를 설정
        self.dataset_configs = dict()
        # 데이터셋 구성을 저장할 딕셔너리 초기화
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        # 데이터 로더에서 사용할 워커 수를 설정
        # num_workers가 제공되지 않으면 기본값으로 배치 사이즈의 두 배를 사용

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
            # `train` 데이터셋 구성 제공 시, 딕셔너리에 저장하고 `_train_dataloader` 메서드를 데이터 로더로 설정

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
            # `validation` 데이터셋 구성 제공 시, 딕셔너리에 저장하고 `_val_dataloader` 메서드를 데이터 로더로 설정

        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
            # `test` 데이터셋 구성 제공 시, 딕셔너리에 저장하고 `_test_dataloader` 메서드를 데이터 로더로 설정

        self.wrap = wrap
        # 데이터셋을 감쌀지 여부를 설정 (True일 경우, WrappedDataset으로 감쌈)

    def prepare_data(self):
        """데이터 다운로드 또는 데이터셋 준비를 위한 메서드"""
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)
            # 데이터셋 구성을 사용하여 데이터셋 인스턴스를 생성
            # `instantiate_from_config` 함수를 통해 데이터셋 객체를 생성

    def setup(self, stage=None):
        """모듈을 설정하고 데이터셋을 초기화하는 메서드"""
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs
        )
        # 데이터셋 구성을 사용하여 각 데이터셋을 인스턴스화하여 `self.datasets` 딕셔너리에 저장

        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
                # `wrap`이 True일 경우, 데이터셋을 `WrappedDataset`으로 감싸서 저장

    def _train_dataloader(self):
        """훈련 데이터 로더를 반환하는 메서드"""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate,
        )
        # 훈련 데이터셋을 위한 DataLoader를 생성하여 반환
        # `batch_size`, `num_workers`, `shuffle` 및 `collate_fn`을 설정

    def _val_dataloader(self):
        """검증 데이터 로더를 반환하는 메서드"""
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )
        # 검증 데이터셋을 위한 DataLoader를 생성하여 반환
        # `batch_size`, `num_workers`, 및 `collate_fn`을 설정

    def _test_dataloader(self):
        """테스트 데이터 로더를 반환하는 메서드"""
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
        )
        # 테스트 데이터셋을 위한 DataLoader를 생성하여 반환
        # `batch_size`, `num_workers`, 및 `collate_fn`을 설정



class SetupCallback(Callback):
    """PyTorch Lightning의 콜백 클래스. 훈련 전 데이터 및 구성 파일을 설정하는 역할"""

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        # 초기화 메서드
        # 콜백의 인스턴스를 초기화하고 필요한 매개변수를 설정
        super().__init__()
        self.resume = resume
        # 체크포인트를 재개하는지 여부를 나타내는 플래그
        self.now = now
        # 현재 시간을 나타내는 문자열 또는 타임스탬프 (파일 이름에 사용)
        self.logdir = logdir
        # 로그 파일을 저장할 디렉토리 경로
        self.ckptdir = ckptdir
        # 체크포인트 파일을 저장할 디렉토리 경로
        self.cfgdir = cfgdir
        # 구성(config) 파일을 저장할 디렉토리 경로
        self.config = config
        # 프로젝트 구성 파일 (OmegaConf 객체)
        self.lightning_config = lightning_config
        # PyTorch Lightning 구성 파일 (OmegaConf 객체)

    def on_pretrain_routine_start(self, trainer, pl_module):
        """훈련 루틴이 시작되기 전에 호출되는 메서드"""
        if trainer.global_rank == 0:
            # `global_rank`가 0인 경우 (주 프로세스에서 실행 중일 때)
            # 로그 디렉토리 및 구성 파일을 생성하고 저장
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            # 로그, 체크포인트, 구성 파일을 저장할 디렉토리 생성

            print("Project config")
            # 프로젝트 구성 출력
            print(self.config.pretty())
            OmegaConf.save(
                self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now))
            )
            # 구성 파일을 지정된 경로에 YAML 형식으로 저장

            print("Lightning config")
            # PyTorch Lightning 구성 출력
            print(self.lightning_config.pretty())
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )
            # PyTorch Lightning 구성 파일을 지정된 경로에 YAML 형식으로 저장

        else:
            # `global_rank`가 0이 아닌 경우 (다른 프로세스에서 실행 중일 때)
            # ModelCheckpoint 콜백이 로그 디렉토리를 생성한 경우 이를 삭제하거나 이동
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                # 로그 디렉토리의 부모 디렉토리와 이름을 분리
                dst = os.path.join(dst, "child_runs", name)
                # 이동할 대상 디렉토리 경로 설정
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                # 대상 디렉토리의 부모 디렉토리 생성

                try:
                    os.rename(self.logdir, dst)
                    # 로그 디렉토리를 새로운 경로로 이동
                except FileNotFoundError:
                    # 파일이 존재하지 않을 경우 예외 처리
                    pass



class ImageLogger(Callback):
    """PyTorch Lightning의 콜백 클래스. 모델의 이미지 출력을 로그에 기록하는 역할"""

    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        # 초기화 메서드
        # 훈련 중 이미지 출력을 기록하는 빈도를 설정
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        # 로깅할 배치 주기 설정
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        # 이미지 값을 -1, 1 범위로 클램프할지 여부를 설정

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        """WandB 로거에 이미지를 기록"""
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)
        # 이미지 그리드를 WandB에 로그로 기록

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        """TestTube 로거에 이미지를 기록"""
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 범위를 0,1 범위로 변환

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
            # 이미지 그리드를 TestTube에 로그로 기록

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """로컬 디렉토리에 이미지를 저장"""
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid + 1.0) / 2.0  # -1,1 범위를 0,1 범위로 변환
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            # 이미지 그리드를 로컬 파일 시스템에 PNG 형식으로 저장

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """훈련 배치 또는 검증 배치에서 이미지를 로깅"""
        if (
            self.check_frequency(batch_idx)
            and hasattr(pl_module, "log_images")  # 배치 주기가 로그 주기에 해당
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
                # 훈련 모드에서 평가 모드로 전환

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)
                # 모델의 `log_images` 메서드를 호출하여 이미지를 가져옴

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)
                        # 이미지 텐서를 -1, 1 범위로 클램프

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )
            # 로컬 디렉토리에 이미지를 저장

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            # 선택된 로거 (WandB 또는 TestTube)에 이미지를 기록

            if is_train:
                pl_module.train()
                # 평가 모드에서 훈련 모드로 복귀

    def check_frequency(self, batch_idx):
        """현재 배치 인덱스가 로깅 주기와 일치하는지 확인"""
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """훈련 배치가 끝날 때 호출되는 메서드"""
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """검증 배치가 끝날 때 호출되는 메서드"""
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    # 현재 시간을 ISO 포맷으로 문자열로 저장
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # 현재 작업 디렉토리를 파이썬 모듈 검색 경로에 추가
    sys.path.append(os.getcwd())

    # 명령행 인자 파서 초기화
    parser = get_parser()
    # PyTorch Lightning의 Trainer 인자를 추가
    parser = Trainer.add_argparse_args(parser)

    # 인자 파싱 (알려지지 않은 인자들도 처리)
    opt, unknown = parser.parse_known_args()
    # --name과 --resume을 동시에 지정하면 오류 발생
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    # --resume이 지정된 경우
    if opt.resume:
        # 지정된 경로가 존재하지 않으면 오류 발생
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        # 경로가 파일인 경우
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # 'logs' 디렉토리의 위치를 찾기
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            # 경로가 디렉토리인 경우
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        # 체크포인트 경로 설정
        opt.resume_from_checkpoint = ckpt
        # 로그 디렉토리의 설정 파일들을 로드
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # 'logs' 디렉토리의 이름을 추출
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        # --name이 지정된 경우
        if opt.name:
            name = "_" + opt.name
        # 설정 파일에서 이름을 추출한 경우
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        # 현재 시간과 이름을 조합하여 로그 디렉토리 이름 설정
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs", nowname)

    # 체크포인트 및 설정 파일 디렉토리 경로 설정
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # 시드를 설정하여 재현성을 확보
    seed_everything(opt.seed)

    try:
        # 설정 파일들을 로드 및 병합
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # Trainer 설정 초기화
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # 기본적으로 분산 학습을 'ddp'로 설정
        trainer_config["distributed_backend"] = "ddp"
        # 커맨드라인 인자와 설정 파일 인자를 병합
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        # GPU 설정 확인 및 CPU 모드 설정
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # 모델 인스턴스화
        model = instantiate_from_config(config.model)

        # 트레이너와 콜백 설정
        trainer_kwargs = dict()

        # 로거 설정
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": "firstTest",
                    "entity": "capstonemz",
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                },
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                },
            },
        }
        # 기본 로거 설정을 병합
        default_logger_cfg = default_logger_cfgs["wandb"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # 모델 체크포인트 설정
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            },
        }
        # 모델에 모니터링할 메트릭이 있는 경우 설정
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        # 체크포인트 설정 병합
        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # 콜백 설정
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
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                },
            },
        }
        # 콜백 설정 병합
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]

        # 트레이너 인스턴스화
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # 데이터 모듈 설정 및 준비
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()

        # 학습률 설정
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )

        # 체크포인트를 저장할 USR1 신호 핸들러 설정
        def melk(*args, **kwargs):
            # 체크포인트 저장
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        # 디버거를 호출할 USR2 신호 핸들러 설정
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        # 신호 핸들러 등록
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # 훈련 및 테스트 실행
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        # 예외 발생 시 디버거 호출
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # 디버그 모드에서 로그 디렉토리를 이동
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
