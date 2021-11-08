from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
from torchvision.datasets.voc import _VOCBase
import torch
import numpy as np


class VOCSegmentation(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_path = self.images[index]
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        metadata = {'id': Path(self.images[index]).stem, 'path': self.images[index], 'shape': (img.size[1], img.size[0])}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, metadata


class VOCSegmentationWithPseudolabels(VOCSegmentation):
    def __init__(self, *args, segments_dir, transform = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_segments_dir(segments_dir)
        self.transform = transform
    
    def _prepare_segments_dir(self, segments_dir):
        self.segments_dir = segments_dir
        # Get segment and image files, which are assumed to be in correspondence
        all_segment_files = sorted(map(str, Path(segments_dir).iterdir()))
        all_img_files = sorted(Path(self.images[0]).parent.iterdir())
        assert len(all_img_files) == len(all_segment_files)
        # Create mapping because I named the segment files badly (sequentially instead of by image id)
        all_img_stems = [p.stem for p in all_img_files]
        valid_img_stems = set([Path(p).stem for p in self.images])  # in our split (e.g. 'val')
        segment_files = []
        for i in range(len(all_img_stems)):
            if all_img_stems[i] in valid_img_stems:
                segment_files.append(all_segment_files[i])
        self.segments = segment_files
        assert len(self.segments) == len(self.images), f'{len(self.segments)=} and {len(self.images)=}'
        print('Loaded segments and images')
        print(f'First image filepath: {self.images[0]}')
        print(f'First segmap filepath: {self.segments[0]}')
        print(f'Last image filepath: {self.images[-1]}')
        print(f'Last segmap filepath: {self.segments[-1]}')

    def _load(self, index: int):
        # Load image
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        target = np.array(Image.open(self.masks[index]))
        metadata = {'id': Path(self.images[index]).stem, 'path': self.images[index], 'shape': tuple(img.shape[:2])}
        # New: load segmap and accompanying metedata
        pseudolabel = np.array(Image.open(self.segments[index]))
        if pseudolabel.size[0] == img.size[0] // 16:  # HACK: this is a hack at the moment
            pseudolabel = pseudolabel.resize(img.size, resample=Image.NEAREST)
        return (img, target, pseudolabel, metadata)

    def __getitem__(self, index: int):
        img, target, pseudolabel, metadata = self._load(index)
        if self.transforms is not None:
            # Transform
            data = self.transform(image=img, mask1=target, mask2=pseudolabel)
            # Unpack
            img, target, pseudolabel = data['image'], data['mask1'], data['mask2']
        return img, target, pseudolabel, metadata


class ContrastiveVOCSegmentationWithPseudolabels(VOCSegmentationWithPseudolabels):
    def __init__(self, *args, segments_dir, transforms_tuple, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_segments_dir(segments_dir)
        assert len(transforms_tuple) == 3
        self.joint_transform = transforms_tuple[0] 
        self.geometric_transform = transforms_tuple[1] 
        self.separate_transform = transforms_tuple[2]

    def __getitem__(self, index: int):
        img, target, pseudolabel, metadata = self._load(index)
        if self.transforms is not None:
            # Join transform
            data = self.joint_transform(image=img, mask1=target, mask2=pseudolabel)
            # Geometric transform
            data = self.geometric_transform(image=data['image'], mask1=data['mask1'], mask2=data['mask2'])
            metadata['replay'] = data['replay']
            # Separate transform
            data = self.separate_transform(image=data['image'], mask1=data['mask1'], mask2=data['mask2'])
            # Unpack
            img, target, pseudolabel = data['image'], data['mask1'], data['mask2']
        return img, target, pseudolabel, metadata
