import os
import glob
import numpy as np
from pathlib import Path
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomFullTrain(Dataset):
    def __init__(
        self,
        image_folder="/opt/ml/DALLE-Couture/data/cropped_img",
        size=256,
    ):

        self.image_path = Path(image_folder)
        self.image_files = [
            *self.image_path.glob("*.png"),
            *self.image_path.glob("*.jpg"),
            *self.image_path.glob("*.jpeg"),
        ]
        self.data = ImagePaths(paths=self.image_files, size=size, random_crop=False)


class CustomFullTest(Dataset):
    def __init__(
        self,
        image_folder="/opt/ml/DALLE-Couture/data/cropped_img",
        size=256,
    ):

        self.image_path = Path(image_folder)
        self.image_files = [
            *self.image_path.glob("*.png"),
            *self.image_path.glob("*.jpg"),
            *self.image_path.glob("*.jpeg"),
        ]
        self.data = ImagePaths(paths=self.image_files, size=size, random_crop=False)
