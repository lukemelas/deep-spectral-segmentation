from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
from pathlib import Path
import torch
import numpy as np
import cv2
import warnings
from torchvision.datasets.voc import VisionDataset, verify_str_arg, DATASET_YEAR_DICT, os 


def _resize_pseudolabel(pseudolabel, img): # HACK HACK HACK
    if (
        (pseudolabel.shape[0] == img.shape[0] // 16) or 
        (pseudolabel.shape[0] == img.shape[0] // 8) or 
        (pseudolabel.shape[0] == 2 * (img.shape[0] // 16))
    ):
        return cv2.resize(pseudolabel, dsize=img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return pseudolabel


class VOCSegmentationWithPseudolabelsBase(VisionDataset):

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            from torchvision.datasets.voc import download_and_extract_archive
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        ######################### NEW #########################
        ######################### NEW #########################
        if self.image_set == 'train':  # everything except val
            image_dir = os.path.join(voc_root, "JPEGImages")
            with open(os.path.join(splits_dir, "val.txt"), "r") as f:
                val_file_stems = set([stem.strip() for stem in f.readlines()])
            all_image_paths = [p for p in Path(image_dir).iterdir()]
            train_image_paths = [str(p) for p in all_image_paths if p.stem not in val_file_stems]
            self.images = sorted(train_image_paths)
            # For the targets, we will just replicate the same target however many times
            target_dir = os.path.join(voc_root, self._TARGET_DIR)
            self.targets = [str(next(Path(target_dir).iterdir()))] * len(self.images)
            
        ######################### END NEW #########################
        ######################### END NEW #########################
        
        else:

            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]

            image_dir = os.path.join(voc_root, "JPEGImages")
            self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

            target_dir = os.path.join(voc_root, self._TARGET_DIR)
            self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

            assert len(self.images) == len(self.targets), ( len(self.images), len(self.targets))

    @property
    def masks(self) -> List[str]:
        return self.targets

    def _prepare_label_map(self, label_map):
        if label_map is not None:
            self.label_map_fn = np.vectorize(label_map.__getitem__)
        else:
            self.label_map_fn = None

    def _prepare_segments_dir(self, segments_dir):
        self.segments_dir = segments_dir
        # Get segment and image files, which are assumed to be in correspondence
        all_segment_files = sorted(map(str, Path(segments_dir).iterdir()))
        all_img_files = sorted(Path(self.images[0]).parent.iterdir())
        assert len(all_img_files) == len(all_segment_files), (len(all_img_files), len(all_segment_files))
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
        pseudolabel = _resize_pseudolabel(pseudolabel, img)  # HACK HACK HACK
        if self.label_map_fn is not None:
            pseudolabel = self.label_map_fn(pseudolabel)
        return (img, target, pseudolabel, metadata)

    def __len__(self) -> int:
        return len(self.images)


class VOCSegmentationWithPseudolabels(VOCSegmentationWithPseudolabelsBase):
    def __init__(self, *args, segments_dir, transform = None, label_map = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_segments_dir(segments_dir)
        self.transform = transform
        self._prepare_label_map(label_map)

    def __getitem__(self, index: int):
        img, target, pseudolabel, metadata = self._load(index)
        if self.transform is not None:
            # Transform
            data = self.transform(image=img, mask1=target, mask2=pseudolabel)
            # Unpack
            img, target, pseudolabel = data['image'], data['mask1'], data['mask2']
        if torch.is_tensor(target):
            target = target.long()
        if torch.is_tensor(pseudolabel):
            pseudolabel = pseudolabel.long()
        return img, target, pseudolabel, metadata


class VOCSegmentationWithPseudolabelsContrastive(VOCSegmentationWithPseudolabelsBase):
    def __init__(self, *args, segments_dir, transforms_tuple, label_map = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_segments_dir(segments_dir)
        assert len(transforms_tuple) == 3
        self.joint_transform = transforms_tuple[0] 
        self.geometric_transform = transforms_tuple[1] 
        self.separate_transform = transforms_tuple[2]
        self._prepare_label_map(label_map)

    def __getitem__(self, index: int):
        img, target, pseudolabel, metadata = self._load(index)
        if self.joint_transform is not None:
            # Join transform
            data_nogeo = self.joint_transform(image=img, mask1=target, mask2=pseudolabel)
            # Geometric transform
            data_geo = self.geometric_transform(image=data_nogeo['image'], mask1=data_nogeo['mask1'], mask2=data_nogeo['mask2'])
            metadata['replay'] = data_geo['replay']
            # Separate transform
            data_nogeo = self.separate_transform(image=data_nogeo['image'], mask1=data_nogeo['mask1'], mask2=data_nogeo['mask2'])
            data_geo = self.separate_transform(image=data_geo['image'], mask1=data_geo['mask1'], mask2=data_geo['mask2'])
            # Unpack
            img_nogeo, target_nogeo, pseudolabel_nogeo = data_nogeo['image'], data_nogeo['mask1'].long(), data_nogeo['mask2'].long()
            img_geo, target_geo, pseudolabel_geo = data_geo['image'], data_geo['mask1'].long(), data_geo['mask2'].long()
        return (img_nogeo, target_nogeo, pseudolabel_nogeo), (img_geo, target_geo, pseudolabel_geo), metadata

        
