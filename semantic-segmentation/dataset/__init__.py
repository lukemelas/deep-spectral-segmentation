import albumentations as A
import albumentations.pytorch as AP
import cv2
from torch.utils.data._utils.collate import default_collate

from .voc import VOCSegmentationWithPseudolabels


def get_transforms(resize_size, crop_size, img_mean, img_std):
    
    # Multiple training transforms for contrastive learning
    train_joint_transform = A.Compose([
        A.SmallestMaxSize(resize_size, interpolation=cv2.INTER_CUBIC),
        A.RandomCrop(crop_size, crop_size),
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
    train_geometric_transform = A.ReplayCompose([
        A.RandomResizedCrop(crop_size, crop_size, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(),
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
    train_separate_transform = A.Compose([
        A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
        A.ToGray(p=0.2), A.GaussianBlur(p=0.1), # A.Solarize(p=0.1)
        A.Normalize(mean=img_mean, std=img_std), AP.ToTensorV2(),
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

    # Validation transform -- no resizing! 
    val_transform = A.Compose([
        # A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC), A.CenterCrop(crop_size, crop_size), 
        A.Normalize(mean=img_mean, std=img_std), AP.ToTensorV2()
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

    train_transforms_tuple = (train_joint_transform, train_geometric_transform, train_separate_transform)
    return train_transforms_tuple, val_transform


def collate_fn(batch):
    everything_but_metadata = [t[:-1] for t in batch]
    metadata = [t[-1] for t in batch]
    return (*default_collate(everything_but_metadata), metadata)


def get_datasets(cfg):

    # Get transforms
    train_transforms_tuple, val_transform = get_transforms(**cfg.data.transform)

    # Get the label map
    if cfg.matching:
        matching = dict(eval(str(cfg.matching)))
        print(f'Using matching: {matching}')
    else:
        matching = None

    # Training dataset
    dataset_train = VOCSegmentationWithPseudolabels(
        **cfg.data.train_kwargs, 
        segments_dir=cfg.segments_dir,
        transforms_tuple=train_transforms_tuple,
        label_map=matching
    )

    # Validation dataset
    dataset_val = VOCSegmentationWithPseudolabels(
        **cfg.data.val_kwargs, 
        segments_dir=cfg.segments_dir,
        transform=val_transform,
        label_map=matching
    )

    return dataset_train, dataset_val, collate_fn
