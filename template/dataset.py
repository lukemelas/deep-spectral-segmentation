import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import (
    default_loader, VisionDataset, make_dataset, IMG_EXTENSIONS, default_loader
)
import cv2


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


if __name__ == "__main__":
    transform = transforms.ToTensor()
    dataset = SimpleDataset(
        root="/home/luke/machine-learning-datasets/image-classification/imagenet/val",
        transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    image, target = next(iter(dataloader))
    print('dataset loading test complete')
    print(image.shape, target.shape)
    import pdb
    pdb.set_trace()
