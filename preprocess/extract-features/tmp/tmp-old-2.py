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
from pymatting import ichol, cg, cf_laplacian, knn_laplacian, normalize_rows, weights_to_laplacian
from skimage.transform import resize
from collections import defaultdict, namedtuple
import scipy.sparse
from scipy.sparse.linalg import eigsh, eigs
import denseCRF
from contexttimer import Timer

# %%

# Load and combine dictionaries
features_dict = torch.load('../features_VOC2007/VOC2007-image-features-dino_vits16-00016.pth')
segments_dict = torch.load('../eigensegments_VOC2007/VOC2007-dino_vits16-eigensegments-00016.pth')  # ./eigensegments
data_dict = {**features_dict, **segments_dict}

# %%

# Sizes
H, W = data_dict['images_resized'][0].shape[-2:]
H_, W_ = H // 16, W // 16
feats = data_dict['k'][0].squeeze() @ data_dict['k'][0].squeeze().T
A = (feats @ feats.T).squeeze()  # affinity_matrix
W_semantic = A.numpy()
W_semantic = (W_semantic - W_semantic.min()) / (W_semantic.max() - W_semantic.min())  # rescale (this does not change anything)
W_semantic_norm = normalize_rows(W_semantic)

# Transform
_inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
])

# Image
from IPython.display import display
image = _inverse_transform(data_dict['images_resized'][0].squeeze(0))
img_np = np.array(image)
img_np_lr = resize(img_np, output_shape=(H_, W_))

display(image)

# %%

# # Affinity
# eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
# eigenvalues, eigenvectors = eigenvalues[:, 0].numpy(), eigenvectors.numpy()
# for k in range(3):
#     print(f'Affinity (torch) {k} ({eigenvalues[k]:.1f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     plt.show()

# Affinity
eigenvalues, eigenvectors = eigsh(W_semantic, k=3, sigma=None, which='LM')
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
for k in range(3):
    print(f'Affinity {k} ({eigenvalues[k]:.1f}):')
    plt.imshow(eigenvectors[:, k].reshape(H_, W_))
    plt.show()

# %%

# # Adding color features -- this is fine, but it doesn't improve results
# W_color = np.linalg.norm(img_np_lr.reshape(H_ * W_, 1, -1) - img_np_lr.reshape(1, H_ * W_, -1), axis=-1)
# lambda_color = 0.2
# _W_semantic = (W_semantic - W_semantic.min()) / (W_semantic.max() - W_semantic.min())
# _W_color = (W_color - W_color.min()) / (W_color.max() - W_color.min())
# W_final = _W_semantic + _W_color * lambda_color
# eigenvalues, eigenvectors = eigsh(W_final, k=3, sigma=None, which='LM')
# eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
# for k in range(3):
#     print(f'Affinity {k} ({eigenvalues[k]:.1f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     plt.show()

# %%
from sklearn.cluster import KMeans

# K-means
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(eigenvectors)
plt.imshow(clusters.reshape(H_, W_))

# %%


# %%

def _rw_laplacian(image, sigma, r):
    h, w = image.shape[:2]
    n = h * w

    m = n * (2 * r + 1) ** 2

    i_inds = np.empty(m, dtype=np.int32)
    j_inds = np.empty(m, dtype=np.int32)
    values = np.empty(m)

    k = 0

    for y in range(h):
        for x in range(w):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x2 = x + dx
                    y2 = y + dy

                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    i = x + y * w
                    j = x2 + y2 * w

                    zi = image[y, x]
                    zj = image[y2, x2]

                    wij = np.exp(-(np.linalg.norm(zi - zj) / sigma) ** 2)

                    i_inds[k] = i
                    j_inds[k] = j

                    values[k] = wij

                    k += 1

    return values, i_inds, j_inds


def rw_affinity(image, sigma=0.033, radius=1, sparse=True):
    import scipy.sparse
    # from pymatting_aot.aot import _rw_laplacian
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W if sparse else W.todense()


def rw_laplacian(image, regularization=1e-8, **kwargs):
    from pymatting.util.util import weights_to_laplacian
    W = rw_affinity(image, **kwargs)
    return weights_to_laplacian(W, regularization=regularization)


def normalize_rows(A, threshold=1e-8, sparse=True):
    row_sums = A.dot(np.ones(A.shape[1], A.dtype))
    row_sums[row_sums < threshold] = 1.0  # prevent division by zero
    row_normalization_factors = 1.0 / row_sums
    diag = scipy.sparse.diags if sparse else np.diag
    print(row_normalization_factors.shape)
    D = diag(row_normalization_factors)
    A = D.dot(A)
    return A


# %%

# # Affinity
# W_semantic = A.numpy()
# W_color = np.asarray(rw_affinity(img_np_lr.astype(np.float32) / 255, sparse=False).astype(np.float32))
# lambda_color = 100
# W_aff = (W_semantic / W_semantic.max()) + (W_color / W_color.max()) * lambda_color
# eigenvalues, eigenvectors = eigsh(W_aff, k=2, sigma=None, which='LM')
# eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
# for k in range(2):
#     print(f'Affinity w/ color {k} ({eigenvalues[k]:.1f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     # plt.imshow(eigenvectors[:, k].reshape(H, W))  # (H_, W_))
#     plt.show()


# %%

# # Object segmentations
# object_eigensegments = data_dict['eigensegments'][0]
# for k, object_eigensegment in range(object_eigensegments):
#     print(f'Eigensegment {k}:')
#     plt.imshow(object_eigensegment)
#     plt.show()

# %%

# Just for fun, let's try out this method: 2006.06573
from numba import njit

# @njit(parallel=True)
def create_g_grid(image: np.array, r: int = 1):
    h, w = image.shape[:2]
    n = h * w
    m = n * (2 * r + 1) ** 2
    i_inds = np.empty(m, dtype=np.int32)
    j_inds = np.empty(m, dtype=np.int32)
    values = np.empty(m)
    k = 0
    for y in range(h):
        for x in range(w):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x2 = x + dx
                    y2 = y + dy
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))
                    i = x + y * w
                    j = x2 + y2 * w
                    # zi = image[y, x]
                    # zj = image[y2, x2]
                    # wij = np.exp(-(np.linalg.norm(zi - zj) / sigma) ** 2)
                    wij = 1  # np.exp(-(np.linalg.norm(zi - zj) / sigma) ** 2)
                    i_inds[k] = i
                    j_inds[k] = j
                    values[k] = wij
                    k += 1
    return values, i_inds, j_inds


# @njit(parallel=True)
def create_g_data(image: np.array, sigma: float = 0.1, patch_size: int = 16):
    assert len(image.shape) == 2, f'{image.shape=}'
    assert image.dtype == np.float32, f'{image.dtype=}'
    P = patch_size
    H, W = image.shape
    H_pad, W_pad = (H // P, W // P)
    image_lr = resize(image, output_shape=(H_pad, W_pad))
    probs = np.power((image_lr.reshape(-1, 1) - image_lr.reshape(1, -1)) / (2 * sigma), 2).reshape(-1)  # prob of picking a pair of patches
    probs = probs / probs.sum()  # normalize probs to sum to 1
    num_samples = 2 * image.size  # m = 2|V|
    image_lr_sampled_indices = np.random.choice(probs.size, size=num_samples, p=probs, replace=False)

    def index_to_pixel(index):
        patch_x, patch_y = index // W_pad, index % W_pad
        x_offset = np.random.randint(0, P)
        y_offset = np.random.randint(0, P)
        x = patch_x * P + x_offset
        y = patch_y * P + y_offset
        return (x, y)
    
    print(f'{image.shape=}')
    print(f'{image_lr.shape=}')
    print(f'{image_lr_sampled_indices.shape=}')
    print(f'{num_samples=}')
    print(f'{probs.shape=}')

    k = 0
    i_inds = np.empty(num_samples, dtype=np.int32)
    j_inds = np.empty(num_samples, dtype=np.int32)
    values = np.empty(num_samples)
    for index_ab in image_lr_sampled_indices:
        index_a = index_ab // (H_pad * W_pad)   # an index from 0 to H_pad*W_pad
        index_b = index_ab % (H_pad * W_pad)   # an index from 0 to H_pad*W_pad
        x_i, y_i = index_to_pixel(index_a)
        x_j, y_j = index_to_pixel(index_b)
        z_i = image[x_i, y_i]
        z_j = image[x_j, y_j]
        w_ij = np.exp(-np.power((z_i - z_j) / sigma, 2) ** 2)
        prob_ab = probs[index_ab]  # this is q(a,b)
        
        # Append
        i_inds[k] = x_i + y_i * W
        j_inds[k] = x_j + y_j * W
        values[k] = w_ij / prob_ab  # equation 13
        k = k + 1
    
    return values, i_inds, j_inds


def create_image_affinity(image, sigma=0.1, radius=1):
    H, W = image.shape[:2]
    # Create
    with Timer(prefix='data'):
        values, i_inds, j_inds = create_g_data(image, sigma, radius)
        W_data = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(H*W, H*W))
    with Timer(prefix='grid'):
        values, i_inds, j_inds = create_g_grid(image, radius)
        W_grid = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(H*W, H*W))
    # Normalize
    with Timer(prefix='normalize'):
        W_data = normalize_rows(W_data)
        W_grid = normalize_rows(W_grid)
    return W_data, W_grid


img_np_lightness = (img_np / 255).mean(axis=-1).astype(np.float32)
with Timer():
    W_data, W_grid = create_image_affinity(img_np_lightness)


# %%
# g_data = create_g_data(img_np_lightness)
# print(f'Created g_data: {g_data}')
# g_grid = create_g_grid(img_np_lightness)
# print(f'Created g_grid: {g_grid}')



