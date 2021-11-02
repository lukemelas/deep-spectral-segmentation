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
from scipy.sparse.linalg import eigsh, eigs
import denseCRF

# %%

# Load
features_dict = torch.load('../features_VOC2007/VOC2007-image-features-dino_vits16-00018.pth')
segments_dict = torch.load('../eigensegments_VOC2007/VOC2007-dino_vits16-eigensegments-00018.pth')  # ./eigensegments
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

# Image
from IPython.display import display
image = _inverse_transform(data_dict['images_resized'][0].squeeze(0))
img_np = np.array(image)
img_np_lr = np.array(image.resize((W_, H_), Image.BILINEAR))

display(image)

# %%

# # Affinity
# eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
# eigenvalues, eigenvectors = eigenvalues[:, 0].numpy(), eigenvectors.numpy()
# for k in range(2):
#     print(f'Affinity (torch) {k} ({eigenvalues[k]:.1f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     plt.show()

# Affinity
_W_semantic = (W_semantic * (W_semantic > 0))
eigenvalues, eigenvectors = eigsh(W_semantic, k=2, sigma=None, which='LM')
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
for k in range(2):
    print(f'Affinity {k} ({eigenvalues[k]:.1f}):')
    plt.imshow(eigenvectors[:, k].reshape(H_, W_))
    plt.show()

# %%

# Laplacian: This works now
# See page 10 of https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf
_W_semantic = (W_semantic * (W_semantic > 0))
_W_semantic = _W_semantic / _W_semantic.max()
_row_sum = _W_semantic @ np.ones(_W_semantic.shape[0])
D = np.diag(_row_sum)  # np.sum(_W_semantic, axis=1))
# D_12 = np.diag(1 / np.sqrt(_row_sum))  # np.sum(_W_semantic, axis=1)))
# L = D_12 @ D_12 @ (D - _W_semantic)
# L = np.eye(_W_semantic.shape[0]) - np.diag(1 / (_W_semantic @ np.ones(_W_semantic.shape[0]))) @ _W_semantic
# eigenvalues, eigenvectors = eigsh(L, k=10, which='SA')
eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=10, which='SA', M=D)
# eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
for k in range(10):
    print(f'Laplacian {k} ({eigenvalues[k]:.3f}):')
    plt.imshow(eigenvectors[:, k].reshape(H_, W_))
    plt.show()

# %%


# %%

# # Laplacian: Why does this not work????
# # D = np.diag(np.sum(W_semantic, axis=1))
# # L = normalize_rows(D - W_semantic, threshold=1e-5)
# # L = weights_to_laplacian(W_semantic, regularization=1e-8)
# _tmp = np.random.randn(H_*W_, 100)
# L = weights_to_laplacian(_tmp @ _tmp.T, regularization=0)
# eigenvalues, eigenvectors = eigs(L, k=3, sigma=0, which='LM')
# eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
# # eigenvalues, eigenvectors = eigsh(L, k=3, which='SM')
# # eigenvalues, eigenvectors = eigsh(L, k=3, sigma=0, which='LM')
# # eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
# for k in range(3):
#     print(f'Laplacian {k} ({eigenvalues[k]:.3f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     plt.show()


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

# Affinity
W_semantic = A.numpy()
W_color = np.asarray(rw_affinity(img_np_lr.astype(np.float32) / 255, sparse=False).astype(np.float32))
lambda_color = 100
W_aff = (W_semantic / W_semantic.max()) + (W_color / W_color.max()) * lambda_color
eigenvalues, eigenvectors = eigsh(W_aff, k=2, sigma=None, which='LM')
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
for k in range(2):
    print(f'Affinity w/ color {k} ({eigenvalues[k]:.1f}):')
    plt.imshow(eigenvectors[:, k].reshape(H_, W_))
    # plt.imshow(eigenvectors[:, k].reshape(H, W))  # (H_, W_))
    plt.show()



# %%

# Object segmentations
object_eigensegments = data_dict['eigensegments'][0]
for k, object_eigensegment in range(object_eigensegments):
    print(f'Eigensegment {k}:')
    plt.imshow(object_eigensegment)
    plt.show()

# %%
