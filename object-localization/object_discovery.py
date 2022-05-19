"""
Main functions for object discovery. 
Code adapted from LOST: https://github.com/valeoai/LOST
"""
from collections import namedtuple
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import scipy
import scipy.ndimage

import numpy as np
from datasets import bbox_iou


def get_eigenvectors_from_features(feats, which_matrix: str = 'affinity_torch', K=2):
    from scipy.sparse.linalg import eigsh

    # Eigenvectors of affinity matrix
    if which_matrix == 'affinity_torch':
        A = feats @ feats.T
        eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
    
    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity':
        A = (feats @ feats.T).cpu().numpy()
        eigenvalues, eigenvectors = eigsh(A, which='LM', k=K)
        eigenvectors = torch.flip(torch.from_numpy(eigenvectors), dims=(-1,))
    
    # Eigenvectors of laplacian matrix
    elif which_matrix == 'laplacian':
        A = (feats @ feats.T).cpu().numpy()
        _W_semantic = (A * (A > 0))
        _W_semantic = _W_semantic / _W_semantic.max()
        diag = _W_semantic @ np.ones(_W_semantic.shape[0])
        diag[diag < 1e-12] = 1.0
        D = np.diag(diag)  # row sum
        try:
            eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=K, sigma=0, which='LM', M=D)
        except:
            eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=K, which='SM', M=D)
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # Eigenvectors of matting laplacian matrix
    elif which_matrix == 'matting_laplacian':

        raise NotImplementedError()

        # # Get sizes
        # B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
        # H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor
        
        # # Load image
        # image_file = str(Path(images_root) / f'{image_id}.jpg')
        # image_lr = Image.open(image_file).resize((W_pad_lr, H_pad_lr), Image.BILINEAR)
        # image_lr = np.array(image_lr) / 255.

        # # Get color affinities
        # W_lr = utils.knn_affinity(image_lr / 255)

        # # Get semantic affinities
        # k_feats_lr = F.interpolate(
        #     k_feats.T.reshape(1, -1, H_patch, W_patch), 
        #     size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
        # ).reshape(-1, H_pad_lr * W_pad_lr).T
        # A_sm_lr = k_feats_lr @ k_feats_lr.T
        # W_sm_lr = (A_sm_lr * (A_sm_lr > 0)).cpu().numpy()
        # W_sm_lr = W_sm_lr / W_sm_lr.max()

        # # Combine
        # W_color = np.array(W_lr.todense().astype(np.float32))
        # W_comb = W_sm_lr + W_color * image_color_lambda  # combination
        # D_comb = utils.get_diagonal(W_comb)

        # # Extract eigenvectors
        # try:
        #     eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
        # except:
        #     eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
        # eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    return eigenvectors


def get_bbox_from_patch_mask(patch_mask, init_image_size, img_np: Optional[np.array] = None):

    # Sizing
    H, W = init_image_size[1:]
    T = patch_mask.numel()
    if (H // 8) * (W // 8) == T:
        P, H_lr, W_lr = (8, H // 8, W // 8)
    elif (H // 16) * (W // 16) == T:
        P, H_lr, W_lr = (16, H // 16, W // 16)
    elif 4 * (H // 16) * (W // 16) == T:
        P, H_lr, W_lr = (8, 2 * (H // 16), 2 * (W // 16))
    elif 16 * (H // 32) * (W // 32) == T:
        P, H_lr, W_lr = (8, 4 * (H // 32), 4 * (W // 32))
    else:
        raise ValueError(f'{init_image_size=}, {patch_mask.shape=}')

    # Create patch mask
    patch_mask = patch_mask.reshape(H_lr, W_lr).cpu().numpy()
    
    # Possibly reverse mask
    # print(np.mean(patch_mask).item())
    if 0.5 < np.mean(patch_mask).item() < 1.0:
        patch_mask = (1 - patch_mask).astype(np.uint8)
    elif np.sum(patch_mask).item() == 0:  # nothing detected at all, so cover the entire image
        patch_mask = (1 - patch_mask).astype(np.uint8) 
    
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
