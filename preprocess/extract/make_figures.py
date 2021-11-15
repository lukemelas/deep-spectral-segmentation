# %% 
import os
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import label2rgb
from matplotlib.cm import get_cmap
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes, make_grid
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
output_file = 'figures/eig-examples-failures-vits16.png'
# eigs_dir = 'matting_laplacian_dino_vitb8_8x_lambda_0' 
# features_dir = 'dino_vitb8'
# image_downsample_factor = 8 
# output_file = 'figures/eig-examples-failures-vitb8.png'

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
image.save(output_file)

# %%

############# Localization figure #############


def get_largest_cc_box(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]


def mask_to_box(patch_mask):

    # Get the box corresponding to the largest connected component of the first eigenvector
    xmin, ymin, xmax, ymax = get_largest_cc_box(patch_mask)
    # pred = [xmin, ymin, xmax, ymax]

    # Rescale to image size
    r_xmin, r_xmax = P * xmin, P * xmax
    r_ymin, r_ymax = P * ymin, P * ymax

    # Prediction bounding box
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    pred[2] = min(pred[2], W)
    pred[3] = min(pred[3], H)

    return np.asarray(pred)


# Get bounding boxes
import pickle
bboxes_root = Path("../../object-localization/outputs/VOC12_train/")
bboxes_dir = "laplacian"
with open(bboxes_root / bboxes_dir / "preds.pkl", "rb") as f:
    preds = pickle.load(f)
with open(bboxes_root / bboxes_dir / "gt.pkl", "rb") as f:
    gt = pickle.load(f)


# %% 

# #  Get inputs
# input_stems = ['2007_000027', '2007_000032', '2007_000033', '2007_000039', '2007_000042', '2007_000061']
input_stems = [f[:-4] for f in list(gt)[2000:2002]]
# # Examples of stems
# input_stems = ['2008_000764', '2008_000705', '2007_000039', '2008_000099', '2008_000499', '2007_009446', '2007_001586']  # example 1  
# input_stems = ['2007_000241', '2007_001586', '2007_001587', '2007_001594', '2007_003020', '2008_000501', '2008_000502']  # example 2
# input_stems = ['2008_000316', '2008_000699', '2007_000033', '2007_000061', '2007_009446',  '2007_000061', '2008_000753']  # example 3 
# input_stems = ['2007_004275', '2007_000032', '2007_000027']
# # Examples of failure cases
# input_stems = ['2007_004289', '2007_004291', '2007_005764', '2007_008085']

# Transform
transform = T.Compose([T.Resize(512), T.ToTensor()])

# Inputs
# Rows with image, ground truth, our prediction
img_tensors = []
for stem in input_stems:
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

    # Get eigenvector: convert to PIL image
    eigenvector = eigenvectors[1].reshape(1, 1, H_pad_lr, W_pad_lr)
    eigenvector = eigenvectors[i].reshape(1, 1, H_pad_lr, W_pad_lr)
    eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
    plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
    eigenvector = Image.open('./tmp.png').convert('RGB') # load back from our temporary location

    # Draw bounding box
    img = (TF.to_tensor(image) * 255).to(torch.uint8)
    img_pred = draw_bounding_boxes(img, boxes=bbox_pred)
    img_gt = draw_bounding_boxes(img, boxes=bbox_gt)
    image_pred = TF.to_pil_image(img_pred)
    image_gt = TF.to_pil_image(img_gt)
    

    # # Get sizes
    # B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    # H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

    # # Add to list
    # img_tensors.append(transform(image))
    # for i in range(1, num_eigs + 1):
    #     eigenvector = eigenvectors[i].reshape(1, 1, H_pad_lr, W_pad_lr)
    #     eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
    #     # eigenvector_colored = binary_colors(255 * eigenvector.squeeze().numpy())[:, :, :3]  # remove alpha channel
    #     # eigenvector_colored = torch.from_numpy(eigenvector_colored).permute(2, 0, 1)
    #     plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
    #     eigenvector = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
    #     img_tensors.append(transform(eigenvector))


# %% 



# %% 

# %% 

# %% 

# %% 

# %% 

# %% 

# %% 

    
#     # Which index
#     which_index = st.number_input(label='Which index to view (0 for all)', value=0)

#     # Load
#     total = 0
#     for i, (image_path, segmap_path) in enumerate(zip(image_paths, segmap_paths)):
#         if total > 40: break
#         image_id = image_path.stem
        
#         # Streamlit
#         cols = []
        
#         # Load
#         image = np.array(Image.open(image_path).convert('RGB'))
#         segmap = np.array(Image.open(segmap_path))
#         segmap_fullres = cv2.resize(segmap, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

#         # Only view images with a specific class
#         if which_index not in np.unique(segmap):
#             continue
#         total += 1

#         # Streamlit
#         cols.append({'image': image, 'caption': image_id})

#         # Load optional bounding boxes
#         bboxes = None
#         if bbox_file is not None:
#             bboxes = torch.tensor(bboxes_list[i]['bboxes_original_resolution'])
#             assert bboxes_list[i]['id'] == image_id, f"{bboxes_list[i]['id']=} but {image_id=}"
#             image_torch = torch.from_numpy(image).permute(2, 0, 1)
#             image_with_boxes_torch = draw_bounding_boxes(image_torch, bboxes)
#             image_with_boxes = image_with_boxes_torch.permute(1, 2, 0).numpy()
            
#             # Streamlit
#             cols.append({'image': image_with_boxes})
            
#         # Color
#         segmap_label_indices, segmap_label_counts = np.unique(segmap, return_counts=True)
#         blank_segmap_overlay = label2rgb(label=segmap_fullres, image=np.full_like(image, 128), 
#             colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=1.0)
#         image_segmap_overlay = label2rgb(label=segmap_fullres, image=image, 
#             colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=0.45)
#         segmap_caption = dict(zip(segmap_label_indices.tolist(), (segmap_label_counts).tolist()))

#         # Streamlit
#         cols.append({'image': blank_segmap_overlay, 'caption': segmap_caption})
#         cols.append({'image': image_segmap_overlay, 'caption': segmap_caption})

#         # Display
#         for d, col in zip(cols, st.columns(len(cols))):
#             col.image(**d)


# if __name__ == '__main__':
#     torch.set_grad_enabled(False)
#     fire.Fire(dict(
#         extract_features=extract_features,  # step 1
#         extract_eigs=extract_eigs,  # step 2
#         # multi-region pipeline
#         extract_multi_region_segmentations=extract_multi_region_segmentations,  # step 3
#         extract_bboxes=extract_bboxes,  # step 4
#         extract_bbox_features=extract_bbox_features,  # step 5
#         extract_bbox_clusters=extract_bbox_clusters,  # step 6
#         extract_semantic_segmentations=extract_semantic_segmentations,  # step 7
#         extract_crf_segmentations=extract_crf_segmentations,  # step 8
#         # single
#         extract_single_region_segmentations=extract_single_region_segmentations, 
#         # vis
#         vis_segmentations=vis_segmentations,  # optional
#     ))