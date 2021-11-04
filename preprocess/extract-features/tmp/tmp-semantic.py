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
features_dict = torch.load('../features_VOC2012/VOC2012-dino_vits16-features-2007_000027.pth')
segments_dict = torch.load('../eigensegments_VOC2012/VOC2012-dino_vits16-eigensegments-2007_000027.pth')  # ./eigensegments
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

plt.rcParams['figure.figsize'] = (H_ // 2, W_ // 2)

def imshow(x):
    plt.imshow(x)
    plt.axis('off')
    plt.show()


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
#     imshow(eigenvectors[:, k].reshape(H_, W_))

# # Affinity
# eigenvalues, eigenvectors = eigsh(W_semantic, k=3, sigma=None, which='LM')
# eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
# for k in range(3):
#     print(f'Affinity {k} ({eigenvalues[k]:.1f}):')
#     imshow(eigenvectors[:, k].reshape(H_, W_))

# Laplacian: This works now
# See page 10 of https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf
_W_semantic = (W_semantic * (W_semantic > 0))
_W_semantic = _W_semantic / _W_semantic.max()
_row_sum = _W_semantic @ np.ones(_W_semantic.shape[0])
D = np.diag(_row_sum)  # np.sum(_W_semantic, axis=1))
eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=5, which='SA', M=D)
for k in range(5):
    print(f'Laplacian {k} ({eigenvalues[k]:.3f}):')
    imshow(eigenvectors[:, k].reshape(H_, W_))

# eigenvectors = data_dict['eigenvectors']
# imshow(eigenvectors.numpy()[0].reshape(H_, W_) > 0)

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
# clusters = kmeans.fit_predict(eigenvectors[:, 1:].numpy().T)
clusters = kmeans.fit_predict(eigenvectors[:, 1:])
imshow(clusters.reshape(H_, W_))

# # Other
# kmeans = AgglomerativeClustering(n_clusters=2)
# clusters = kmeans.fit_predict(eigenvectors.numpy().T)
# imshow(clusters.reshape(H_, W_))

# %%
plt.imshow(np.array(Image.fromarray(clusters.reshape(H_, W_)).convert('L')))

# %%

import denseCRF

H_patch, W_patch = H_, W_
H_pad, W_pad = img_np.shape[:2]

# Params
ParamsCRF = namedtuple('ParamsCRF', 'w1 alpha beta w2 gamma it')
CRF_PARAMS = ParamsCRF(
    w1    = 20,     # weight of bilateral term  # 10.0,
    alpha = 30,    # spatial std  # 80,  
    beta  = 13,    # rgb  std  # 13,  
    w2    = 5,     # weight of spatial term  # 3.0, 
    gamma = 3,     # spatial std  # 3,   
    it    = 5.0,   # iteration  # 5.0, 
)

# CRF
U = F.interpolate(
    torch.from_numpy(clusters).reshape(1, 1, H_patch, W_patch).to(torch.uint8),
    size=(H_pad, W_pad), mode='nearest'
).squeeze()
U = F.one_hot(U.long(), num_classes=n_clusters) * 1.0 + 0.0 / n_clusters
eigensegment = denseCRF.densecrf(img_np, U, CRF_PARAMS)
imshow(eigensegment)


# %%


# %%
