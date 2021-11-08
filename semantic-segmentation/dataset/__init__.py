import albumentations as A
import albumentations.pytorch as AP
import cv2

from .voc import VOCSegmentationWithPseudolabels, ContrastiveVOCSegmentationWithPseudolabels


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
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

    # Validation transform -- no resizing! 
    val_transform = A.Compose([
        # A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC), A.CenterCrop(crop_size, crop_size), 
        AP.ToTensor(), A.Normalize(mean=img_mean, std=img_std)
    ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})

    train_transforms_tuple = (train_joint_transform, train_geometric_transform, train_separate_transform)
    return train_transforms_tuple, val_transform


def get_datasets(cfg):

    # Get transforms
    train_transforms_tuple, val_transform = get_transforms(**cfg.data.transform)

    # Training dataset
    dataset_train = ContrastiveVOCSegmentationWithPseudolabels(
        **cfg.data.train_kwargs, 
        segments_dir=cfg.segments_dir,
        transforms_tuple=train_transforms_tuple,
    )

    # Validation dataset
    dataset_val = VOCSegmentationWithPseudolabels(
        **cfg.data.val_kwargs, 
        segments_dir=cfg.segments_dir,
        transform=val_transform,
    )

    return dataset_train, dataset_val


# def _test():
#     # TODO: make this work
#     from torch.utils.data import DataLoader
#     from matplotlib.cm import get_cmap
#     from skimage.color import label2rgb
#     import numpy as np
#     import streamlit as st

#     # Create dataset
#     dataset = VOCSegmentationWithPseudolabels(
#         root="/data_q1_d/machine-learning-datasets/semantic-segmentation/PASCAL_VOC/VOC2012", 
#         segments_dir="/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/crf/fixed/segmaps_e2_d8_pca_32",
#         year="2012", image_set="val", transform=None)

#     # Visualize ground truth
#     colors = get_cmap('tab20', 21).colors[:,:3]
#     for i, (image, target, mask, metadata) in enumerate(dataset):
#         if i >= 3: break
#         image = np.array(image)
#         target = np.array(target)
#         target[target == 255] = 0  # set the "unknown" regions to background for visualization
#         # Overlay mask on image
#         image_pred_overlay = label2rgb(label=mask, image=image, colors=colors[:1], bg_label=0, alpha=0.45)
#         image_target_overlay = label2rgb(label=target, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
#         # Display
#         cols = st.columns(4)
#         cols[0].image(image, caption=metadata['id'])
#         cols[1].image(target, caption=metadata['id'])
#         cols[2].image(image_target_overlay, caption=metadata['id'])
#         cols[3].image(image_pred_overlay, caption=metadata['id'])
    
#     # import pdb
#     # pdb.set_trace()


# if __name__ == "__main__":
#     _test()
