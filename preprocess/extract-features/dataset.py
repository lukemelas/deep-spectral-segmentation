from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def albumentations_to_torch(transform):
    def _transform(image):
        return transform(image=image)["image"]
    return _transform


def get_transforms(resize_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = A.Compose([
        A.RandomResizedCrop(crop_size, crop_size), A.HorizontalFlip(),
        A.Normalize(mean=mean, std=std), ToTensorV2()])
    val_transform = A.Compose([
        A.SmallestMaxSize(resize_size), A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=mean, std=std), ToTensorV2()])
    return albumentations_to_torch(train_transform), albumentations_to_torch(val_transform)


class ImagesDataset(Dataset):
    def __init__(self, filenames: str, images_root: Optional[str] = None, transform: Optional[Callable] = None,
                 prepare_filenames: bool = True) -> None:
        self.root = None if images_root is None else Path(images_root)
        self.filenames = sorted(list(set(filenames))) if prepare_filenames else filenames
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.filenames[index]
        full_path = path if self.root is None else str(self.root / path)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, path, index

    def __len__(self) -> int:
        return len(self.filenames)
