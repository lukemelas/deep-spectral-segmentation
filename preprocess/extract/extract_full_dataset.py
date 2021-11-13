from PIL import Image
from accelerate import Accelerator
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from scipy.sparse.linalg import eigsh
from skimage.measure import label as measure_label
from skimage.measure import perimeter as measure_perimeter
from skimage.morphology import binary_erosion, binary_dilation
from skimage.transform import resize
from torch.functional import _return_counts
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from typing import Callable, Iterable, List, Optional, Tuple, Union
from typing import Optional
import cv2
import denseCRF
import fire
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans, DBSCAN

import extract_utils as utils



def stack_features(
    features_dir: str,
    output_dir: str,
):
    """
    Example:
        python extract_full_dataset.py stack_features \
            --features_dir "./data/VOC2012/features" \
            --output_dir "./data/VOC2012/full_dataset"
    """
    utils.make_output_dir(output_dir, check_if_empty=True)

    # Process
    index = 0 
    all_indices = []  # contains the start index for every image
    all_features = []  # contains the features
    for i, p in enumerate(tqdm(list(Path(features_dir).iterdir()))):
        data_dict = torch.load(p, map_location='cpu')
        feats = data_dict['k'].squeeze(0)  # (T, D)
        all_indices.append(index)
        index += feats.shape[0]
        all_features.append(feats)
    print('Loaded features. Stacking...')
    all_features = torch.cat(all_features, dim=0)
    all_indices.append(index)  # add the last index
    assert all_indices[-1] == len(all_features)

    # Save
    output_dir = Path(output_dir)
    print('Finished stacking. Saving...')
    torch.save(all_features, str(output_dir / 'features.pth'))
    torch.save(all_indices, str(output_dir / 'indices.pth'))
    print(f'Saved features and indices to {output_dir}')


def perform_svd(
    features_dir: str,
    output_dir: str,
):
    """
    Example:
        python extract_full_dataset.py perform_svd \
            --features_dir "./data/VOC2012/full_dataset" \
            --output_dir "./data/VOC2012/full_dataset"
    """
    utils.make_output_dir(output_dir, check_if_empty=False)

    # Dask
    import dask 
    import dask.array as da
    # import cupy  # it would be cool to use CUDA
    # from dask_cuda import LocalCluster
    
    # See https://blog.dask.org/2020/05/13/large-svds
    dask.config.set({"optimization.fuse.ave-width": 5})

    # Load
    feats = torch.load(Path(features_dir) / 'features.pth')  # (NT, D)
    print(f'Loaded features with shape {feats.shape=}')

    # Dask
    feats = da.from_array(feats.numpy(), chunks=20_000)  # convert to chunked dask array
    u, s, v = da.linalg.svd_compressed(feats, k=21, compute=True)
    u, s, v = np.array(u), np.array(s), np.array(v)
    print(u.shape)
    print(s.shape)
    print(v.shape)
    
    # Save
    output_dir = Path(output_dir)
    print('Finished SVD. Saving...')
    torch.save((u, s, v), str(output_dir / 'svd.pth'))
    print(f'Saved (U, S, V) to {output_dir}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        stack_features=stack_features,
        perform_svd=perform_svd
    ))