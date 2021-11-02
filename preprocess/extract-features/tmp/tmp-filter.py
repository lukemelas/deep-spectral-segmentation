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

features_root = './features'
segments_root = './object_eigensegments'

# Load
feature_files = sorted(Path(features_root).iterdir())[25:50]
segment_files = sorted(Path(segments_root).iterdir())[25:50]
assert len(feature_files) == len(segment_files)

# Combine
combined_output_dict = defaultdict(list)
for i, (ff, fs) in enumerate(list(zip(feature_files, segment_files))):
    features_dict = torch.load(ff, map_location='cpu')
    segments_dict = torch.load(fs, map_location='cpu')
    for k, v in features_dict.items():
        combined_output_dict[k].append(v)
    for k, v in segments_dict.items():
        combined_output_dict[k].append(v)
data_dict = combined_output_dict = dict(combined_output_dict)

# %%

# Image index
image_idx = 17

# Sizes
H, W = data_dict['images_resized'][image_idx].shape[-2:]
H_, W_ = H // 16, W // 16
feats = data_dict['k'][image_idx].squeeze() @ data_dict['k'][image_idx].squeeze().T
A = (feats @ feats.T).squeeze()  # affinity_matrix
W_semantic = A.numpy()

# Transform
_inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
])

# Image
from IPython.display import display
image = _inverse_transform(data_dict['images_resized'][image_idx].squeeze(0))
img_np = np.array(image)
img_np_lr = np.array(image.resize((W_, H_), Image.BILINEAR))

display(image)

# %%

# Object segmentations
object_eigensegments = data_dict['eigensegments'][image_idx]
plt.imshow(object_eigensegments[0])
plt.show()

# %%

# Affinity
eigenvalues, eigenvectors = eigsh(W_semantic, k=2, sigma=None, which='LM')
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
# for k in range(2):
#     print(f'Affinity {k} ({eigenvalues[k]:.1f}):')
#     plt.imshow(eigenvectors[:, k].reshape(H_, W_))
#     plt.show()

device = 'cuda'
ParamsCRF = namedtuple('ParamsCRF', 'w1 alpha beta w2 gamma it')
crf_params = ParamsCRF(
    w1    = 6,     # weight of bilateral term  # 10.0,
    alpha = 40,    # spatial std  # 80,  
    beta  = 13,    # rgb  std  # 13,  
    w2    = 3,     # weight of spatial term  # 3.0, 
    gamma = 3,     # spatial std  # 3,   
    it    = 10.0,   # iteration  # 5.0, 
)

def get_largest_cc(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc_mask = (labels == largest_cc_index)
    return largest_cc_mask

k = 0
H_patch, W_patch = H_, W_
H_pad, W_pad = H, W
threshold = 0
eigenvectors = torch.from_numpy(np.ascontiguousarray(eigenvectors))
eigensegment = eigenvectors[:, k].numpy()
eigensegment = (eigensegment > threshold).astype(np.uint8)
if 0.5 < np.mean(eigensegment).item() < 1.0:
    eigensegment = (1 - eigensegment)

# Do CRF
unary_potentials = F.interpolate(torch.from_numpy(eigensegment).reshape(1, 1, H_patch, W_patch).float(), size=(H_pad, W_pad), mode='bilinear').squeeze()
unary_potentials_np = np.stack((1 - unary_potentials, unary_potentials), axis=-1)
eigensegment = denseCRF.densecrf(img_np, unary_potentials_np, crf_params)

# Get largest connected component
eigensegment = get_largest_cc(eigensegment)

plt.imshow(eigensegment)
plt.show()

# %%

# %%
