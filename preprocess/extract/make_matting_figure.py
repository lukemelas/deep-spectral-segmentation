# %% 
import os
import sys
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from scipy.ndimage import interpolation
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import scipy
from skimage.morphology import binary_erosion, binary_dilation
from pymatting import (
    ichol, cg, cf_laplacian, knn_laplacian, normalize_rows, weights_to_laplacian, 
    lkm_laplacian, stack_images, estimate_alpha_cf
)
from pymatting.laplacian.rw_laplacian import _rw_laplacian
from skimage.transform import resize
from collections import defaultdict, namedtuple
import scipy.sparse
from scipy.sparse.linalg import eigsh, eigs
import cv2
from torchvision.utils import make_grid

import extract_utils as utils

# %%

stems = ['COCO_train2014_000000001025', 'COCO_train2014_000000000382', 'COCO_train2014_000000000625', 'COCO_train2014_000000000036', 'COCO_train2014_000000005903']
stem = stems[0]
image_downsample_factor = 8

# Load
image_file = f'tmp/{stem}.jpg'
eig_file = f'tmp/{stem}.pth'

# Load
image = Image.open(image_file)
img = np.array(image) / 255
data_dict = torch.load(eig_file)

# Sign ambiguity
eigenvectors = data_dict['eigenvectors']
for k in range(eigenvectors.shape[0]):
    if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
        eigenvectors[k] = 0 - eigenvectors[k]

# Resize eigenvector
eigenvector = eigenvectors[1].reshape(img.shape[0] // 8, img.shape[1] // 8)


# # Get sizes
# B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
# H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

# # Add to list
# eigenvector_images.append(transform(image))
# for i in range(1, num_eigs + 1):
#     eigenvector = eigenvectors[i].reshape(1, 1, H_pad_lr, W_pad_lr)
#     eigenvector = F.interpolate(eigenvector, size=(H, W), mode='nearest')  # slightly off, but for visualizations this is okay
#     plt.imsave('./tmp.png', eigenvector.squeeze().numpy())  # save to a temporary location
#     eigenvector = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
#     eigenvector_images.append(transform(eigenvector))



# %% 
# # Eignvector
eigenvector = eig['eigenvectors'][0].reshape(img.shape[0] // 8, img.shape[1] // 8)



# %% 
# %% 
# %% 
# %% 
# %% 

# # Load
# images_root = "./data/VOC2012/images"
# data_dict = torch.load("./data/VOC2012/features/dino_vitb16/2007_000068.pth")  # 2007_000123
# image_id = data_dict['file'][:-4]

# # Parameter
# K = 6  # number of eigenvectors

# # %%

# # Load image
# image_pil = Image.open(f'{images_root}/{image_id}.jpg')
# image = np.array(image_pil)
# print(f'{image.shape=}')

# # Load affinity matrix
# k_feats = data_dict['k'].squeeze()
# A = k_feats @ k_feats.T
# print(f'{A.shape=}')

# # Truncate image
# B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
# image = image[:H_pad, :W_pad]
# print(f'{image.shape=}')

# # Display
# plt.rcParams['figure.figsize'] = (H_patch // 2, W_patch // 2)
# def imshow(x): plt.imshow(x); plt.axis('off'); plt.show()

# # Show image
# imshow(image)

# # %% 

# # # Let's now try combining the semantic features with the color features
# # # First, we have to get a sense of how well semantic features do on their own

# # Perhaps we should try on a smaller resolution, say downsampled 4x (so 256x)?
# lr_factor = 8  # 4
# H_pad_lr, W_pad_lr = H_pad // lr_factor, W_pad // lr_factor
# image_lr = np.array(Image.fromarray(image).resize((W_pad_lr, H_pad_lr), Image.BILINEAR))
# print(f'{image_lr.shape=}')

# # Low resolution laplacian
# # W_lr = utils.knn_affinity(image_lr / 255)
# W_lr = utils.rw_affinity(image_lr / 255)
# D_lr = utils.get_diagonal(W_lr)
# print(f'{W_lr.shape=} and {D_lr.shape=}')

# # Get eigenvectors
# eigenvalues, eigenvectors = eigsh(D_lr - W_lr, k=K, sigma=0, which='LM', M=D_lr)
# print('Color')
# imshow(np.hstack([
#     eigenvectors[:, k].reshape(H_pad_lr, W_pad_lr)
#     for k in range(K)
# ]))

# # %%

# # Let's now do the features alone
# W_sm = (A * (A > 0)).cpu().numpy()
# W_sm = W_sm / W_sm.max()
# D_sm = utils.get_diagonal(W_sm)

# # Get eigenvectors
# eigenvalues, eigenvectors = eigsh(D_sm - W_sm, k=K, sigma=0, which='LM', M=D_sm)
# print('Features')
# imshow(np.hstack([
#     eigenvectors[:, k].reshape(H_patch, W_patch)
#     for k in range(K)
# ]))

# # %%

# # Now, we are going to do the same thing, but at a higher resolution
# k_feats_lr = F.interpolate(
#     k_feats.T.reshape(1, -1, H_patch, W_patch), 
#     size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
# ).reshape(-1, H_pad_lr * W_pad_lr).T
# A_sm_lr = k_feats_lr @ k_feats_lr.T
# W_sm_lr = (A_sm_lr * (A_sm_lr > 0)).cpu().numpy()
# W_sm_lr = W_sm_lr / W_sm_lr.max()
# D_sm_lr = utils.get_diagonal(W_sm_lr)

# # Get eigenvectors
# print('Features (higher resolution)')
# eigenvalues, eigenvectors = eigsh(D_sm_lr - W_sm_lr, k=K, sigma=0, which='LM', M=D_sm_lr)
# imshow(np.hstack([
#     eigenvectors[:, k].reshape(H_pad_lr, W_pad_lr)
#     for k in range(K)
# ]))


# # %%

# # Finally, we're going to combine our methods
# lambda_color = 100.0
# W_color = np.array(W_lr.todense().astype(np.float32))
# W_comb = W_sm_lr + W_color * lambda_color  # combination
# D_comb = utils.get_diagonal(W_comb)

# # Get eigenvectors
# print(f'Combined with {lambda_color=}')
# eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
# imshow(np.hstack([
#     eigenvectors[:, k].reshape(H_pad_lr, W_pad_lr)
#     for k in range(K)
# ]))


# # %%

# # # Now, we are going to do the same thing as above, but sparse,
# # # where we get sparsity by enforcing only local connections. 
# # # It turns out, this is actually quite terrible.
# # k_feats_lr = F.interpolate(
# #     k_feats.cuda().T.reshape(1, -1, H_patch, W_patch), 
# #     size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
# # ).reshape(-1, H_pad_lr * W_pad_lr).T
# # A_sm_lr = k_feats_lr @ k_feats_lr.T
# # W_sm_lr = (A_sm_lr * (A_sm_lr > 0))  # .cpu().numpy()
# # W_sm_lr = W_sm_lr / W_sm_lr.max()
# # # W_sm_lr = np.triu(np.tril(W_sm_lr, k=128), k=-128)  # local connections
# # W_sm_lr = W_sm_lr  #  * (torch.rand_like(W_sm_lr) < 1/1000)
# # W_sm_lr = scipy.sparse.csr_matrix(W_sm_lr.cpu().numpy())

# # lambda_color = 1
# # W_color = W_lr
# # W_comb = W_sm_lr + W_color * lambda_color  # combination
# # D_comb = utils.get_diagonal(W_comb)

# # # Get eigenvectors
# # print('Starting...')
# # eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
# # for k in range(K):
# #     imshow(eigenvectors[:, k].reshape(H_pad_lr, W_pad_lr))

# # %%
