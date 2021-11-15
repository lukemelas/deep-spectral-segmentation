from collections import namedtuple
from typing import Tuple
import torch
import torch.nn.functional as F
import scipy
import scipy.ndimage

import numpy as np
from datasets import bbox_iou

# NEW: CRF -- https://github.com/HiLab-git/SimpleCRF
ParamsCRF = namedtuple('ParamsCRF', 'w1 alpha beta w2 gamma it')
crf_params = ParamsCRF(
    w1    = 6,     # weight of bilateral term  # 10.0,
    alpha = 40,    # spatial std  # 80,  
    beta  = 13,    # rgb  std  # 13,  
    w2    = 3,     # weight of spatial term  # 3.0, 
    gamma = 3,     # spatial std  # 3,   
    it    = 5.0,   # iteration  # 5.0, 
)


def apply_crf(unary_potentials_low_res: torch.Tensor, img_np: np.array, dims: Tuple):
    import denseCRF

    H, W = img_np.shape[:2]
    H_, W_ = dims

    # Get unary potentials
    unary_potentials = F.interpolate(unary_potentials_low_res.reshape(1,1,H_,W_), size=(H,W), mode='bilinear').squeeze()
    unary_potentials = (unary_potentials - unary_potentials.min()) / (unary_potentials.max() - unary_potentials.min())
    unary_potentials_np = torch.stack((1 - unary_potentials, unary_potentials), dim=-1).cpu().numpy()
    
    # Return result of CRF
    out = denseCRF.densecrf(img_np, unary_potentials_np, crf_params)
    return out





def eigen_lost(feats, dims, scales, init_image_size, k_patches=100, img_np = None):
    initial_im_size=init_image_size[1:]
    w_featmap, h_featmap = dims

    # Affinity matrix
    A = (feats @ feats.transpose(1, 2)).squeeze()

    # Compute the similarity
    eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
    patch_mask = eigenvectors[:, 0] > 0  # <-- this is the approximate mask
    patch_mask = patch_mask.reshape(w_featmap, h_featmap)
    patch_mask = patch_mask.cpu().numpy()

    # # LOST: This gets the reported 61.44 performance
    # sorted_patches, scores = patch_scoring(A)
    # seed = sorted_patches[0]
    # potentials = sorted_patches[:k_patches]
    # similars = potentials[A[seed, potentials] > 0.0]
    # M = torch.sum(A[similars, :], dim=0)
    # patch_mask = (M > 0).reshape(w_featmap, h_featmap).cpu().numpy()
    
    # # CRF or rescale
    # patch_mask = apply_crf(M, img_np, dims)
    # H, W = img_np.shape[:2]
    # H_, W_ = dims
    # patch_mask = (F.interpolate(M.reshape(1, 1, H_, W_), size=(H, W), mode='bilinear').squeeze() > 0).cpu().numpy()
    
    # Possibly reverse mask
    # print(np.mean(patch_mask).item())
    if 0.5 < np.mean(patch_mask).item() < 1.0:
        patch_mask = (1 - patch_mask).astype(np.uint8)
    elif np.sum(patch_mask).item() == 0:  # nothing detected at all
        patch_mask = (1 - patch_mask).astype(np.uint8) 
    
    # Get the box corresponding to the largest connected component of the first eigenvector
    xmin, ymin, xmax, ymax = get_largest_cc_box(patch_mask)
    # pred = [xmin, ymin, xmax, ymax]

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    # Prediction bounding box
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    return np.asarray(pred)


def lost(feats, dims, scales, init_image_size, k_patches=100):
    """
    Implementation of LOST method.
    Inputs
        feats: the pixel/patche features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
    Outputs
        pred: box predictions
        A: binary affinity matrix
        scores: lowest degree scores for all patches
        seed: selected patch corresponding to an object
    """
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()

    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)

    # Select the initial seed
    seed = sorted_patches[0]

    # Seed expansion
    potentials = sorted_patches[:k_patches]
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[similars, :], dim=0)

    # Box extraction
    pred, _ = detect_box(
        M, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )

    return np.asarray(pred), A, M, scores, seed


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()  # NOTE: this was not used. should this be used?

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    # Should not happen with LOST
    if cc == 0:
        raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(labeled_array == cc)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats


def dino_seg(attn, dims, patch_size, head=0):
    """
    Extraction of boxes based on the DINO segmentation method proposed in https://github.com/facebookresearch/dino. 
    """
    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    # We keep only the output patch attention
    # Get the attentions corresponding to [CLS] token
    attentions = attn[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Connected components
    labeled_array, num_features = scipy.ndimage.label(th_attn[head].cpu().numpy())

    # Find the biggest component
    size_components = [np.sum(labeled_array == c) for c in range(np.max(labeled_array))]

    if len(size_components) > 1:
        # Select the biggest component avoiding component 0 corresponding to background
        biggest_component = np.argmax(size_components[1:]) + 1
    else:
        # Cases of a single component
        biggest_component = 0

    # Mask corresponding to connected component
    mask = np.where(labeled_array == biggest_component)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image
    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    return pred


def get_largest_cc_box(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]
