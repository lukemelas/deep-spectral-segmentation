import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from omegaconf.dictconfig import DictConfig
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, VisionDataset, make_dataset, IMG_EXTENSIONS
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg: DictConfig):
    crop_size, resize_size = cfg.data.transform.crop_size, cfg.data.transform.resize_size
    train_transform = A.Compose([
        A.RandomResizedCrop(crop_size, crop_size),
        A.HorizontalFlip(),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2()])
    val_transform = A.Compose([
        A.Resize(resize_size, resize_size),
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2()])
    return train_transform, val_transform


class SimpleDataset(VisionDataset):
    """ A simple version of the PyTorch ImageFolder dataset with 
        albumentations augmentations """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        classes, class_to_idx = self._find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions=IMG_EXTENSIONS)
        assert len(self.samples) > 0, f"Found 0 files in subfolders of: {self.root}\n"
        self.loader = default_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            tsfm_dict = self.transform(image=image)
            image = tsfm_dict["image"]
        return image, target

    def __len__(self) -> int:
        return len(self.samples)


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, transform=None) -> None:
        super().__init__()
        self.datasets = datasets
        self.transform = transform
        self.length = len(datasets[0])
        for dataset in datasets:
            assert len(dataset) <= self.length, 'different sized datasets'

    def __getitem__(self, index):
        samples = [dataset[index] for dataset in self.datasets]
        if self.transform is not None:
            samples = self.transform(*samples)
        return samples

    def __len__(self):
        return self.length


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    val_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
    dataset_1 = SimpleDataset(
        root="/home/luke/machine-learning-datasets/image-classification/imagenet/val",
        transform=val_transform)
    dataset_2 = SimpleDataset(
        root="/home/luke/machine-learning-datasets/image-classification/imagenet/val",
        transform=val_transform)
    dataset = ZipDataset([dataset_1, dataset_2])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    (image_1, target_1), (image_2, target_2) = next(iter(dataloader))
    print('dataset loading test complete')
    print(image_1.shape, target_1.shape, image_2.shape, target_2.shape)
    import pdb
    pdb.set_trace()
