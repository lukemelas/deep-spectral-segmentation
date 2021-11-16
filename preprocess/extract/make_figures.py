# %% 
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
from PIL import Image
import numpy as np
import math
import torch
import torch.nn.functional as F
from skimage.color import label2rgb
from skimage.measure import label as measure_label
from matplotlib.cm import get_cmap
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.ops import box_iou
import matplotlib.pyplot as plt

import extract_utils as utils

# Global paths
images_list = Path("./data/VOC2012/lists/images.txt")
images_root = Path("./data/VOC2012/images")
eigs_root = Path("./data/VOC2012/eigs")
features_root = Path("./data/VOC2012/features")

# Prepare
images_list = images_list.read_text().splitlines()

# Colors
colors = get_cmap('tab20', 21).colors[:, :3]
binary_colors = get_cmap('viridis')

# Specific paths
eigs_dir = 'laplacian'
features_dir = 'dino_vits16'
image_downsample_factor = 16
output_file_eig = 'figures/eig-examples-failures-vits16.png'
# eigs_dir = 'matting_laplacian_dino_vitb8_8x_lambda_0' 
# features_dir = 'dino_vitb8'
# image_downsample_factor = 8 
# output_file_eig = 'figures/eig-examples-failures-vitb8.png'

# %% 

############# Eigensegments figure #############

# Get inputs
# input_stems = ['2007_000027', '2007_000032', '2007_000033', '2007_000039', '2007_000042', '2007_000061']
# input_stems = [f[:-4] for f in images_list[1100:1135]]
# # Examples of stems
# input_stems = ['2008_000764', '2008_000705', '2007_000039', '2008_000099', '2008_000499', '2007_009446', '2007_001586']  # example 1  
# input_stems = ['2007_000241', '2007_001586', '2007_001587', '2007_001594', '2007_003020', '2008_000501', '2008_000502']  # example 2
# input_stems = ['2008_000316', '2008_000699', '2007_000033', '2007_000061', '2007_009446',  '2007_000061', '2008_000753']  # example 3 
# input_stems = ['2007_004275', '2007_000032', '2007_000027']
# # Examples of failure cases
input_stems = ['2007_004289', '2007_004291', '2007_005764', '2007_008085']

# Transform
transform = T.Compose([T.Resize(512), T.CenterCrop(512), T.ToTensor()])

# Inputs
num_eigs = 4
nrow = 1 * (4 + 1)  # number of images per row
img_tensors = []
for stem in input_stems:
    image_file = images_root / f'{stem}.jpg'
    segmap_file = eigs_root / eigs_dir / f'{stem}.pth'
    features_file = features_root / features_dir / f'{stem}.pth'

    # Load 
    image = Image.open(image_file)
    data_dict = {}
    data_dict.update(torch.load(features_file))
    data_dict.update(torch.load(segmap_file))

    # Sign ambiguity
    eigenvectors = data_dict['eigenvectors']
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
    
    # Get sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

    # Add to list
    img_tensors.append(transform(image))
    for i in range(1, num_eigs + 1):
        eigenvector = eigenvectors[i].reshape(1, 1, H_pad_lr, W_pad_lr)
        eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
        # eigenvector_colored = binary_colors(255 * eigenvector.squeeze().numpy())[:, :, :3]  # remove alpha channel
        # eigenvector_colored = torch.from_numpy(eigenvector_colored).permute(2, 0, 1)
        plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
        eigenvector = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
        img_tensors.append(transform(eigenvector))

# Stack 
img_tensor_grid = make_grid(img_tensors, nrow=nrow, pad_value=1)
image = TF.to_pil_image(img_tensor_grid)
image.save(output_file_eig)
print(f'Saved to {output_file_eig}')

# %%

############# Localization figure #############


def get_largest_cc(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_cc_index


# Get bounding boxes
bboxes_root = Path("../../object-localization/outputs/VOC12_train/")
bboxes_dir = "laplacian"
output_file_loc = "./figures/loc-examples-failuers-vits16.png"

# Load
import pickle
with open(bboxes_root / bboxes_dir / "preds.pkl", "rb") as f:
    preds = pickle.load(f)
with open(bboxes_root / bboxes_dir / "gt.pkl", "rb") as f:
    gt = pickle.load(f)

#  Get inputs (2300 or 2200 good I think, though not sure)
# input_stems = [f[:-4] for f in list(gt)[1000:]]
# input_stems = [f[:-4] for f in list(gt)[1500:]]
# input_stems = [f[:-4] for f in list(gt)[2000:]]
# input_stems = [f[:-4] for f in list(gt)[2500:]]
# input_stems = [f[:-4] for f in list(gt)[1337:]]
# # Examples of failure cases
# input_stems = ['2007_004289', '2007_004291', '2007_005764', '2007_008085']

def resize_width(x: Image):
    w, h = x.size
    x = x.resize((384, (h * 384) // w))
    return TF.to_tensor(x)

# Inputs
# Rows with image, ground truth, our prediction
np_rows = []
for stem in input_stems:
    if len(np_rows) >= 12: break  # end 

    # Get paths
    image_file = images_root / f'{stem}.jpg'
    segmap_file = eigs_root / eigs_dir / f'{stem}.pth'
    features_file = features_root / features_dir / f'{stem}.pth'

    # Load 
    image = Image.open(image_file)
    bbox_gt = torch.from_numpy(gt[f'{stem}.jpg']).reshape(-1, 4)
    bbox_pred = torch.from_numpy(preds[f'{stem}.jpg']).reshape(-1, 4)
    data_dict = {}
    data_dict.update(torch.load(features_file))
    data_dict.update(torch.load(segmap_file))

    # Get eigenvector: sign ambiguity
    eigenvectors = data_dict['eigenvectors']
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
    
    # Get eigenvector: get sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor
    if H_pad_lr > W_pad_lr:
        continue

    # Get eigenvector: largest connected component
    eigenvector = eigenvectors[1].reshape(1, 1, H_pad_lr, W_pad_lr)
    eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
    eigenvector = eigenvector.squeeze().numpy()
    eigenvector = get_largest_cc(eigenvector > 0)  # this or just the eigenvector? not sure, let's try this for now

    # Get eigenvector: overlay on image to PIL image
    colors = ['royalblue']  # ['tomato']
    # colors = np.array([[0.1607843137254902, 0.6862745098039216, 0.4980392156862745]])
    img_region_overlay = label2rgb(label=eigenvector, image=np.array(image), bg_label=0, colors=colors, alpha=0.35)
    img_region_overlay = (torch.from_numpy(img_region_overlay).permute(2, 0, 1) * 255).to(torch.uint8)
    image_region_overlay = TF.to_pil_image(img_region_overlay)

    # Check
    max_iou = box_iou(bbox_gt, bbox_pred).max().item()
    pred_color = 'limegreen' if (max_iou > 0.5) else 'orangered'

    # Generate only bad failure cases
    if (max_iou > 0.3):
        continue

    # Draw bounding box
    img = (TF.to_tensor(image) * 255).to(torch.uint8)
    img_pred = draw_bounding_boxes(img, boxes=bbox_pred, width=4, colors=[pred_color] * 10)
    img_gt = draw_bounding_boxes(img, boxes=bbox_gt, width=4, colors=['lightseagreen'] * 10)
    image_pred = TF.to_pil_image(img_pred)
    image_gt = TF.to_pil_image(img_gt)

    # Transform
    tensors_row = list(map(resize_width, [image, image_region_overlay, image_pred, image_gt]))
    tensors_row = make_grid(tensors_row, nrow=4, pad_value=1)
    np_rows.append(np.array(TF.to_pil_image(tensors_row)))
    np_rows.append(np.ones((2, 1546, 3), dtype=np.uint8) * 255)  # padding

# Stack
np_grid = np.vstack(np_rows)
image_grid = Image.fromarray(np_grid)
image_grid.save(output_file_loc)
print(f'Saved to {output_file_loc}')

# %% 
image_grid


# %% 
