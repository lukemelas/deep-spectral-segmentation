# %%

import os
import sys
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from IPython.display import display
import matplotlib.pyplot as plt
import scipy
from skimage.morphology import binary_erosion, binary_dilation
from pymatting import ichol, cg, cf_laplacian, knn_laplacian, lkm_laplacian, blend, stack_images
from pymatting.laplacian.knn_laplacian import normalize_rows, knn
from skimage.transform import resize
from contexttimer import Timer
from scipy.sparse.linalg import eigsh


# %%

# Paths
info = torch.load("./tmp/info.pth", map_location='cpu')

# %%

# Transform after model (for visualization or if an image is produced)
inverse_transform = T.Compose([ # T.Resize(288), T.CenterCrop(224),
    T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    T.ToPILImage()
])

# %%

# Unpack
feats = info['feats'].squeeze()
img = info['img']
scores = info['scores']

# Params
k_patches = 100
H, W = img.shape[-2:]
H_, W_ = H // 16, W // 16

# Get the affinity matrix
A = (feats @ feats.T).squeeze()  # affinity_matrix

# Get the 
sorted_patches = torch.argsort(scores, descending=True)

# Select the initial seed
seed = sorted_patches[0]
assert seed == info['seed']  # check

# Seed expansion
potentials = sorted_patches[:k_patches]
similars = potentials[A[seed, potentials] > 0.0]
M = torch.sum(A[similars, :], dim=0)  # a rough segmentation mask
mask = (M > 0)

# Example of eigenvectors
with Timer(prefix='Calculating eigenvalues'):
    eigenvalues, eigenvectors = eigsh(A.numpy(), k=6, which='LM')  # shift-invert mode
for k in range(6):
    plt.imshow(eigenvectors[:, k].real.reshape(H_, W_))  # (H, W))
    plt.show()

# %%

# Show image
display(inverse_transform(img))

# %%

# image = load_image('../examples/VOC07_000236_modified.jpg')
np_image = np.array(inverse_transform(img.cpu())) / 255
L = cf_laplacian(np_image)

# %%

def knn_matrix(
    image,
    n_neighbors=[20, 10],
    distance_weights=[2.0, 0.1],
    which='laplacian'
):
    """
    This function calculates the KNN matting Laplacian matrix as described in :cite:`chen2013knn`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    n_neighbors: list of ints
        Number of neighbors to consider. If :code:`len(n_neighbors)>1` multiple nearest neighbor calculations are done and merged, defaults to `[20, 10]`, i.e. first 20 neighbors are considered and in the second run :math:`10` neighbors. The pixel distances are then weighted by the :code:`distance_weights`.
    distance_weights: list of floats
        Weight of distance in feature vector, defaults to `[2.0, 0.1]`.

    Returns
    ---------
    L: scipy.sparse.spmatrix
        Matting Laplacian matrix
    """
    assert which in ['laplacian', 'affinity']
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    if which == 'affinity':
        return W
    else:
        W = normalize_rows(W)
        I = scipy.sparse.identity(n)
        L = I - W
        return L

L_knn = knn_matrix(np_image, which='laplacian')
A_knn = knn_matrix(np_image, which='affinity')

# %% 

# KNN laplacian eigenvalues
from scipy.sparse.linalg import eigsh
with Timer(prefix='Calculating laplacian eigenvalues'):
    eigenvalues, eigenvectors = eigsh(L_knn, k=6, which='SM')
for k in range(6):
    plt.imshow(eigenvectors[:, k].real.reshape(H, W))
    plt.show()

color_only_evecs = eigenvectors

# %%

from scipy.sparse.linalg import LinearOperator

feats_fullres = F.interpolate(feats.transpose(0, 1).reshape(1, -1, H_, W_), size=(H, W), mode='bilinear')
feats_fullres_flat = feats_fullres.reshape(-1, H*W).transpose(0, 1)  # (HW, D)
feats_fullres_flat_np = feats_fullres_flat.numpy()
feats_fullres_flat_np = (feats_fullres_flat_np - feats_fullres_flat_np.min()) / (feats_fullres_flat_np.max() - feats_fullres_flat_np.min())
A_knn_norm = normalize_rows(A_knn)

# %%
laplacian_kwargs = {}
L_lkm, _ = lkm_laplacian(np_image, **laplacian_kwargs)

# %%

A_numpy = A.numpy()
def matvec(x):  # 168960
    # Color
    out_knn = A_knn_norm * x
    # out_cf = L * x
    # # Semantic (full resolution)
    # out_semantic = feats_fullres_flat_np @ (feats_fullres_flat_np.T @ x)
    # Semantic (low resolution)
    x_small = F.interpolate(torch.from_numpy(x).float().reshape(1, 1, H, W), size=(H_, W_), mode='bilinear').reshape(H_ * W_)
    out_semantic = x_small @ A
    out_semantic = F.interpolate(out_semantic.reshape(1, 1, H_, W_), size=(H, W), mode='bilinear').reshape(H * W).numpy()
    return out_knn + out_semantic * 1e-6
    # return out_semantic
    # return x @ A_numpy

LaplacianLinearOperator = LinearOperator((H*W, H*W), matvec=matvec)
# LaplacianLinearOperator = LinearOperator((H_*W_, H_*W_), matvec=matvec)

K = 10
with Timer(prefix='Calculating laplacian eigenvalues'):
    # eigenvalues, eigenvectors = eigsh(LaplacianLinearOperator, k=K, which='LM', sigma=0)  # , tol=1e-2)  # shift-invert mode
    eigenvalues, eigenvectors = eigsh(LaplacianLinearOperator, k=K, which='LM')  # , tol=1e-3)
    # eigenvalues, eigenvectors = eigsh(A.numpy(), k=K, which='LM')  # , tol=1e-3)
for k in range(3):
    plt.imshow(eigenvectors[:, K - k - 1].real.reshape(H, W))
    plt.show()

# %%

from PIL import Image
from IPython.display import display

# Color (eigenvectors with only color)
_color_only_evecs = [e.reshape(H, W) for e in color_only_evecs.T]
_tmp = []
for x in _color_only_evecs[:3]:
    plt.imsave('./tmp.png', x)  # save to a temporary location
    x = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
    _tmp.append(np.array(x))
_tmp = np.hstack(_tmp)
image_grid_color_only = Image.fromarray(_tmp)

# Color (eigenvectors with features)
_with_affinity_evecs = [e.reshape(H, W) for e in eigenvectors.T][::-1]
_tmp = []
for x in _with_affinity_evecs[:3]:
    plt.imsave('./tmp.png', x)  # save to a temporary location
    x = Image.open('./tmp.png').convert('RGB') # load back from our temporary location
    _tmp.append(np.array(x))
_tmp = np.hstack(_tmp)
image_grid_with_affinity = Image.fromarray(_tmp)

# Save
image_grid_color_only.save('figures/matting-color_only.png')
image_grid_with_affinity.save('figures/matting-with_semantics.png')

# Save actual image
image = inverse_transform(img)
image.save('figures/matting-image.png')

# %%

# Image of eigenvector
evec = _with_affinity_evecs[0]
alpha = (evec - evec.min()) / (evec.max() - evec.min())
Image.fromarray((alpha * 255).astype(np.uint8)).save('figures/matting-alpha.png')

# Example of cutout of image
cutout = stack_images(np_image, alpha)
cutout_image = Image.fromarray((cutout * 255).astype(np.uint8))
cutout_image.save('figures/matting-cutout_image.png')

# Composite image onto background 
bg = np.kron([[1, 0] * 16, [0, 1] * 16] * 16, np.ones((16, 16)))
# Image.fromarray((128 + np.ones_like(np_image) * 127 * bg[:np_image.shape[0], :np_image.shape[1], None]).astype(np.uint8))
bg = (128 + np.ones_like(np_image) * 127 * bg[:np_image.shape[0], :np_image.shape[1], None]) / 255
composite_image = blend(np_image, bg, np.clip(alpha * 1.6, 0, 1))
display(Image.fromarray((composite_image * 255).astype(np.uint8)) ) # .save('figures/matting-composite.png')
