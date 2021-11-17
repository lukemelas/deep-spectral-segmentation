# %% 
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
from PIL import Image
import numpy as np
import math
import torch
from torch.functional import _return_counts
import torch.nn.functional as F
from skimage.color import label2rgb
from skimage.measure import label as measure_label
from matplotlib.cm import get_cmap
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from IPython.display import display

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
    eigs_file = eigs_root / eigs_dir / f'{stem}.pth'
    features_file = features_root / features_dir / f'{stem}.pth'

    # Load 
    image = Image.open(image_file)
    data_dict = {}
    data_dict.update(torch.load(features_file))
    data_dict.update(torch.load(eigs_file))

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
        plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
        eigenvector = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
        img_tensors.append(transform(eigenvector))

# Stack 
img_tensor_grid = make_grid(img_tensors, nrow=nrow, pad_value=1)
image = TF.to_pil_image(img_tensor_grid)
# image.save(output_file_eig)
# print(f'Saved to {output_file_eig}')

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
output_file_loc = "./figures/loc-examples-failures-vits16.png"

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
# input_stems = [f[:-4] for f in list(gt)[2200:]]  # examples-6
# input_stems = [f[:-4] for f in list(gt)[2500:]]
input_stems = [f[:-4] for f in list(gt)[420:]]
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
    eigs_file = eigs_root / eigs_dir / f'{stem}.pth'
    features_file = features_root / features_dir / f'{stem}.pth'

    # Load 
    image = Image.open(image_file)
    bbox_gt = torch.from_numpy(gt[f'{stem}.jpg']).reshape(-1, 4)
    bbox_pred = torch.from_numpy(preds[f'{stem}.jpg']).reshape(-1, 4)
    data_dict = {}
    data_dict.update(torch.load(features_file))
    data_dict.update(torch.load(eigs_file))

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
    img_pred = draw_bounding_boxes(img, boxes=bbox_pred, width=10, colors=[pred_color] * 10)
    img_gt = draw_bounding_boxes(img, boxes=bbox_gt, width=10, colors=['lightseagreen'] * 10)
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
# image_grid.save(output_file_loc)
# print(f'Saved to {output_file_loc}')
display(image_grid)
print(output_file_loc)

# %% 
image_grid


# %% 

############# Images for diagram #############

# input_stems = ['2010_005906']  # ['2008_000316', '2008_000699', '2007_000033', '2007_000061', '2007_009446',  '2007_000061', '2008_000753']  # example 3 
input_stems = ['2007_004275']

# New paths
segmap_dir = 'laplacian_dino_vitb16_fixed_15'  # 'laplacian_dino_vitb8_fixed_15'
semseg_run = 'segmaps_e_d5_pca_0_s12'

# Colors
colors = get_cmap('tab20', 21).colors[:, :3][::-1]

# Show images
for stem in input_stems:
    image_file = images_root / f'{stem}.jpg'
    eigs_file = eigs_root / eigs_dir / f'{stem}.pth'
    features_file = features_root / features_dir / f'{stem}.pth'
    seg_file = features_root / '..' / 'multi_region_segmentation' / segmap_dir / f'{stem}.png'
    # TODO: Make sure 
    semseg_file = features_root / '..' / 'semantic_segmentations' / 'patches' / segmap_dir / semseg_run / f'{stem}.png'

    # Load 
    image = Image.open(image_file)
    data_dict = {}
    data_dict.update(torch.load(features_file))
    data_dict.update(torch.load(eigs_file))

    # New transform
    transform = T.Compose([T.Resize(512), T.CenterCrop(512)])

    # Sign ambiguity
    eigenvectors = data_dict['eigenvectors']
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
    
    # Get sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

    # Add to list
    image = transform(image)
    image.save(f'figures/method-diagram-{stem}-image.png')
    for i in range(1, 3):
        eigenvector = eigenvectors[i].reshape(1, 1, H_pad_lr, W_pad_lr)
        eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
        plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
        eigenvector_image = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
        transform(eigenvector_image).save(f'figures/method-diagram-{stem}-evec-{i}.png')
    
    # Get eigenvector: largest connected component
    mask = F.interpolate(eigenvectors[1].reshape(1, 1, H_pad_lr, W_pad_lr), size=(H, W), mode='nearest')
    mask = TF.center_crop(TF.resize(mask, 512, interpolation=TF.InterpolationMode.NEAREST), 512)
    mask = (mask.squeeze().numpy() > 0)
    mask_largest_cc = get_largest_cc(mask)  # this or just the eigenvector? not sure, let's try this for now
    Image.fromarray(mask).save(f'figures/method-diagram-{stem}-mask.png')
    Image.fromarray(mask_largest_cc).save(f'figures/method-diagram-{stem}-mask-cc.png')

    # Get bounding box
    where_mask = np.where(mask_largest_cc)
    ymin, ymax = min(where_mask[0]), max(where_mask[0]) + 1
    xmin, xmax = min(where_mask[1]), max(where_mask[1]) + 1
    boxes = [[xmin, ymin, xmax, ymax]]

    # Bounding boxes
    img = (TF.to_tensor(image) * 255).to(torch.uint8)
    img_pred = draw_bounding_boxes(img, boxes=torch.tensor(boxes), width=10, colors=['limegreen'])
    image_pred = TF.to_pil_image(img_pred)
    image_pred.save(f'figures/method-diagram-{stem}-bbox.png')

    # Segmentations
    segmap = np.array(TF.center_crop(TF.resize(Image.open(seg_file), 512, interpolation=TF.InterpolationMode.NEAREST), 512))
    semseg = np.array(TF.center_crop(TF.resize(Image.open(semseg_file), 512, interpolation=TF.InterpolationMode.NEAREST), 512))

    # Color
    segmap_label_indices, segmap_label_counts = np.unique(segmap, return_counts=True)
    semseg_label_indices, semseg_label_counts = np.unique(semseg, return_counts=True)
    blank_segmap_overlay = label2rgb(label=segmap, image=np.full_like(image, 128), colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=1.0)
    blank_semseg_overlay = label2rgb(label=semseg, image=np.full_like(image, 128), colors=colors[semseg_label_indices[semseg_label_indices != 0]], bg_label=0, alpha=1.0)
    image_segmap_overlay = label2rgb(label=segmap, image=np.array(image), colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=0.8)
    image_semseg_overlay = label2rgb(label=semseg, image=np.array(image), colors=colors[semseg_label_indices[semseg_label_indices != 0]], bg_label=0, alpha=0.8)
    segmap_image = Image.fromarray((image_segmap_overlay * 255).astype(np.uint8))
    semseg_image = Image.fromarray((image_semseg_overlay * 255).astype(np.uint8))

    # Save
    segmap_image.save(f'figures/method-diagram-{stem}-segmap.png')
    semseg_image.save(f'figures/method-diagram-{stem}-semseg.png')

print('Done')

# %% 
############# Segmentation figure #############


############# Images for diagram #############

# input_stems = ['2007_000033', '2007_000042', '2007_000061', '2007_000123', '2007_000175']  # , '2007_000187', '2007_000323', '2007_000332', '2007_000346']
# input_stems = ['2011_001775','2011_002247','2011_001292','2007_009794','2009_003756','2010_005108','2011_002575','2009_000716','2007_000559','2009_004497']
# input_stems = ['2008_006063', '2010_001579', '2007_001763', '2008_008051', '2009_004882', '2007_003188', '2009_002221', '2007_007165', '2009_000771', '2011_001988']
input_stems = ['2007_000042', '2007_000061', '2007_000123', '2007_001763', '2007_001884']
# input_stems = ['2009_001768', '2010_001376', '2009_003059', '2010_002142', '2007_000727', '2007_003088', '2010_001327', '2011_000566', '2010_003947', '2007_001299', '2007_006086', '2010_004559', '2007_008964', '2007_001884', '2007_008547', '2008_007194', '2009_003323', '2010_005664', '2007_007688', '2009_000488']
# input_stems = input_stems[15:20]

# input_stems = ['2007_000033', '2007_000042', '2007_001763', '2009_004882']


# New paths
preds_dir = Path('/data_q1_d/extra-storage/found_new/outputs/generate/2021-11-17--00-10-22/preds')
gt_dir = Path('/data_q1_d/extra-storage/found_new/outputs/generate/2021-11-17--00-10-22/gt')
mc_dir = Path('/data_q1_d/extra-storage/found_new/outputs/scp_from_maskcontrast')

# Colors
colors = get_cmap('tab20', 21).colors[:, :3][::-1]

# Show images
nrow = 4
img_tensors = []
for stem in input_stems:
    image = Image.open(images_root / f'{stem}.jpg')
    pred_image = Image.open(preds_dir / f'{stem}.png')
    gt_image = Image.open(gt_dir / f'{stem}.png')
    mc_image = np.load(mc_dir / f'{stem}.npy').astype(np.uint8)  # maskcontrast image
    mc_image = Image.fromarray(mc_image).resize(image.size, resample=Image.NEAREST)
    assert image.size == pred_image.size == gt_image.size, (image.size, pred_image.size, gt_image.size)
    assert image.size == mc_image.size, (image.size, mc_image.size)

    # Pred and ground truth
    pred = np.array(pred_image)
    gt = np.array(gt_image)
    mc = np.array(mc_image)

    # Unknown region --> 0
    pred[pred == 255] = 0
    gt[gt == 255] = 0

    # Color
    pred_label_indices, pred_label_counts = np.unique(pred, return_counts=True)
    gt_label_indices, gt_label_counts = np.unique(gt, return_counts=True)
    mc_label_indices, mc_label_counts = np.unique(mc, return_counts=True)
    # 
    blank_pred_overlay = label2rgb(label=pred, image=np.full_like(image, 128), colors=colors[pred_label_indices[pred_label_indices != 0]], bg_label=0, alpha=1.0)
    blank_gt_overlay = label2rgb(label=gt, image=np.full_like(image, 128), colors=colors[gt_label_indices[gt_label_indices != 0]], bg_label=0, alpha=1.0)
    blank_mc_overlay = label2rgb(label=mc, image=np.full_like(image, 128), colors=colors[mc_label_indices[mc_label_indices != 0]], bg_label=0, alpha=1.0)
    # 
    image_pred_overlay = label2rgb(label=pred, image=np.array(image), colors=colors[pred_label_indices[pred_label_indices != 0]], bg_label=0, alpha=0.8)
    image_gt_overlay = label2rgb(label=gt, image=np.array(image), colors=colors[gt_label_indices[gt_label_indices != 0]], bg_label=0, alpha=0.8)
    image_mc_overlay = label2rgb(label=mc, image=np.array(image), colors=colors[pred_label_indices[pred_label_indices != 0]], bg_label=0, alpha=0.8)
    # 
    pred_image = Image.fromarray((image_pred_overlay * 255).astype(np.uint8))
    gt_image = Image.fromarray((image_gt_overlay * 255).astype(np.uint8))
    mc_image = Image.fromarray((image_mc_overlay * 255).astype(np.uint8))

    # Torch
    for img in [image, mc_image, pred_image, gt_image]:
        img_tensors.append(TF.to_tensor(TF.center_crop(TF.resize(img, size=384, interpolation=TF.InterpolationMode.NEAREST), 384)))

# Stack
output_file_semseg = 'figures/semseg-comparison.png'
img_tensor_grid = make_grid(img_tensors, nrow=nrow, pad_value=1)
image_grid = TF.to_pil_image(img_tensor_grid)
image_grid.save(output_file_semseg)
print(f'Saved to {output_file_semseg}')
display(image_grid)
print('Done')



# %% 
############# Matting figure #############

# %% 

# # Quick script 
# from tqdm import tqdm
# nrow = 3
# img_tensors = []
# pbar = tqdm(preds_dir.iterdir())
# nseg = 0
# _files = []
# for i, p in enumerate(pbar):
#     _pred, counts = np.unique(np.array(Image.open(p)), return_counts=True)
#     nseg += len(pred)
#     if len(counts) > 2:
#         _files.append((p.name, counts[2].item(), counts, _pred))
#     if i % 10 == 0:
#         pbar.set_description(f"{i/nseg}")

# _sorted_files = sorted(_files, key=lambda x: -x[2].min())[:100]

# %%
