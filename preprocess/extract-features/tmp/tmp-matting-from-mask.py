# %% 

import os
import sys
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import scipy
from skimage.morphology import binary_erosion, binary_dilation
from pymatting import ichol, cg, cf_laplacian, knn_laplacian, normalize_rows, weights_to_laplacian, lkm_laplacian, stack_images, estimate_alpha_cf
from skimage.transform import resize
from collections import defaultdict, namedtuple
from scipy.sparse.linalg import eigsh, eigs
import denseCRF

# %%

# Load
features_dict = torch.load('../features_VOC2012/VOC2012-image-features-dino_vits16-00157.pth')
segments_dict = torch.load('../eigensegments_VOC2012/VOC2012-dino_vits16-eigensegments-00157.pth')
combined_dict = defaultdict(list)
for k, v in features_dict.items():
    combined_dict[k].append(v)
for k, v in segments_dict.items():
    combined_dict[k].append(v)
data_dict = combined_dict = dict(combined_dict)

# %%

# Sizes
H, W = data_dict['images_resized'][0].shape[-2:]
H_, W_ = H // 16, W // 16
feats = data_dict['k'][0].squeeze() @ data_dict['k'][0].squeeze().T
A = (feats @ feats.T).squeeze()  # affinity_matrix
W_semantic = A.numpy()

# Transform
_inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
])

plt.rcParams['figure.figsize'] = (H_ // 2, W_ // 2)

def imshow(x):
    plt.imshow(x)
    plt.axis('off')
    plt.show()


# Image
from IPython.display import display
image = _inverse_transform(data_dict['images_resized'][0].squeeze(0))
img_np = np.array(image)
img_np_lr = np.array(image.resize((W_, H_), Image.BILINEAR))
mask = data_dict['eigensegments_object'][0][0]
img_np_float = img_np / 255

imshow(img_np)
imshow(mask)


# %%

def erode_or_dilate_mask(x: Union[torch.Tensor, np.ndarray], r: int = 0, erode=True):
    fn = binary_erosion if erode else binary_dilation
    for _ in range(r):
        x = fn(x)
    return x

# %%

eigenvector = data_dict['eigenvectors'][0][0].reshape(H_,W_)
eigenvector = resize(eigenvector, output_shape=(H, W))  # upscale
object_segment = (eigenvector > 0).astype(float)
if 0.5 < np.mean(object_segment).item() < 1.0:  # reverse segment
    object_segment = (1 - object_segment)

def trimap_from_mask(mask, r=15):
    is_fg = erode_or_dilate_mask(mask, r=r, erode=True)
    is_bg = ~(erode_or_dilate_mask(mask, r=r, erode=False))
    if is_fg.sum() == 0:
        is_fg = erode_or_dilate_mask(mask, r=1, erode=True)
    trimap = np.full_like(mask, fill_value=0.5, dtype=float)
    trimap[is_fg] = 1.0
    trimap[is_bg] = 0.0
    return trimap

trimap = trimap_from_mask(object_segment)

# from pymatting import estimate_alpha_cf, estimate_alpha_knn, estimate_alpha_lkm, estimate_alpha_lbdm, estimate_foreground_ml

# alpha = estimate_alpha_cf(img_np_float, trimap)

imshow(trimap)
# imshow(alpha)
# imshow(stack_images(img_np_float, alpha))


# %%

def trimap_from_mask(mask, r=15):
    is_fg = erode_or_dilate_mask(mask, r=r, erode=True)
    is_bg = ~(erode_or_dilate_mask(mask, r=r, erode=False))
    trimap = np.full_like(mask, fill_value=0.5, dtype=float)
    trimap[is_fg] = 1.0
    trimap[is_bg] = 0.0
    return trimap

trimap = trimap_from_mask(mask)

from pymatting import estimate_alpha_cf, estimate_alpha_knn, estimate_alpha_lkm, estimate_alpha_lbdm, estimate_foreground_ml

alpha = estimate_alpha_cf(img_np_float, trimap)

imshow(alpha)

imshow(stack_images(img_np_float, alpha))


# %%


def perform_matting(img: np.array, patch_mask: np.array, L = None):
    """ Performs alpha matting given a preliminary patch mask and an image """
    
    # Reshape mask to 2D
    H, W, _ = img.shape
    # patch_mask = patch_mask.reshape(H_, W_)

    # Get matting laplacian
    if L is None:
        L = cf_laplacian(img)  # cf_laplacian(img)

    # Create trimap
    r = 25
    is_fg = erode_or_dilate_mask(patch_mask, r=r, erode=True)
    is_bg = ~(erode_or_dilate_mask(patch_mask, r=r, erode=False))
    is_fg = resize(is_fg, output_shape=(H, W)).astype(bool).reshape(-1)
    is_bg = resize(is_bg, output_shape=(H, W)).astype(bool).reshape(-1)
    is_known = (is_fg | is_bg)
    
    # # Visualize
    # plt.imshow(is_known.reshape((H, W)))
    # plt.show()
    
    # Linear system
    lambda_value = 100
    C_m = scipy.sparse.diags(lambda_value * is_known)
    A_m = L + C_m  # matting affinity matrix

    # Build ichol preconditioner for faster convergence
    A_m = A_m.tocsr()
    A_m.sum_duplicates()
    M_m = ichol(A_m)

    # Solve linear system
    b = (lambda_value * is_fg).astype(np.float64)
    x = cg(A_m, b, M=M_m)

    # # From pymatting/alpha/estimate_alpha_cf.py
    # is_unknown = ~is_known
    # L_U = L[is_unknown, :][:, is_unknown]
    # R = L[is_unknown, :][:, is_known]
    # m = is_fg[is_known]
    # x = is_fg.copy().ravel()
    # x[is_unknown] = cg(L_U, -R.dot(m), M=ichol(L_U))

    # Result
    alpha = np.clip(x, 0.0, 1.0).reshape(H, W)
    return alpha

# %% 

alpha = perform_matting(img=(img_np / 255), patch_mask=mask)
plt.imshow(alpha)
plt.show()
plt.imshow(stack_images((img_np / 255), alpha))
plt.show()
# foreground = estimate_foreground_ml((img_np / 255), alpha, return_background=False)
# plt.imshow(foreground)
# plt.show()

# %%
for i in range(20,24):
    tmp = torch.load(f"../mattes_VOC2012/VOC2012-dino_vits16-matte-000{i}.pth")
    imshow(tmp[:,:,:3])
    imshow(tmp[:,:,3])
    imshow(tmp)

# %%
