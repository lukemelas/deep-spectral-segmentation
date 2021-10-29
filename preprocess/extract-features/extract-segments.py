"""An experimental script to create eigensegments"""
import time
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from functools import partial
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import fire
from collections import defaultdict, namedtuple
import denseCRF
from scipy.sparse.linalg import eigsh


# Inverse transform
_inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])

# Params
ParamsCRF = namedtuple('ParamsCRF', 'w1 alpha beta w2 gamma it')
CRF_PARAMS = ParamsCRF(
    w1    = 6,     # weight of bilateral term  # 10.0,
    alpha = 40,    # spatial std  # 80,  
    beta  = 13,    # rgb  std  # 13,  
    w2    = 3,     # weight of spatial term  # 3.0, 
    gamma = 3,     # spatial std  # 3,   
    it    = 5.0,   # iteration  # 5.0, 
)

def get_largest_cc(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc_mask = (labels == largest_cc_index)
    return largest_cc_mask


def _create_object_segment(
    inp: Tuple[int, str], K: int, threshold: float, crf_params: Tuple, 
    prefix: str, output_dir: str, patch_size: int = 16
):
    # Load 
    index, path = inp
    data_dict = torch.load(path, map_location='cpu')

    # Sizes
    P = patch_size
    B, C, H, W = data_dict['shape']
    assert B == 1, 'assumption violated :('
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P
    k_feats = data_dict['k'].squeeze()  # CPU
    img_np = np.array(_inverse_transform(data_dict['images_resized'].squeeze(0)))

    # Upscale features
    features = F.interpolate(
        data_dict['out'][0][:, 1:].permute(2, 0, 1).reshape(1, -1, H_patch, W_patch), 
        size=(H_pad, W_pad), mode='bilinear', align_corners=False
    ).squeeze()

    # # Eigenvectors of affinity matrix
    # A = k_feats @ k_feats.T
    # eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
    
    # Eigenvectors of affinity matrix with scipy
    from scipy.sparse.linalg import eigsh
    A = k_feats @ k_feats.T
    eigenvalues, eigenvectors = eigsh(A.cpu().numpy(), which='LM', k=K)  # find small eigenvalues
    eigenvectors = torch.flip(torch.from_numpy(eigenvectors), dims=(-1,))
    
    # # Eigenvectors of laplacian matrix
    # from scipy.sparse.csgraph import laplacian
    # from scipy.sparse.linalg import eigsh
    # A = (k_feats @ k_feats.T).cpu().numpy()
    # L = laplacian(A, normed=False)
    # eigenvalues, eigenvectors = eigsh(L, sigma=0, which='LM', k=K)  # find small eigenvalues
    # eigenvectors = torch.from_numpy(eigenvectors)
    # print(eigenvectors.shape)

    # CRF
    new_data_dict = defaultdict(list)
    for k in range(K):
        eigenvector = eigenvectors[:, k]

        ############# Segments ############
        eigensegment = eigenvector.clone()
        if 0.5 < torch.mean((eigensegment > threshold).float()).item() < 1.0:  # reverse segment
            eigensegment = 0 - eigensegment
        # Do CRF on high-resolution features
        U = F.interpolate(
            eigensegment.reshape(1, 1, H_patch, W_patch), 
            size=(H_pad, W_pad), mode='bilinear', align_corners=False
        ).squeeze()
        U = (U - U.min()) / (U.max() - U.min())
        U = torch.stack((1 - U, U), dim=-1)
        eigensegment = denseCRF.densecrf(img_np, U.numpy(), crf_params)
        eigensegment = torch.from_numpy(eigensegment)
        # Get feature
        pooled_feature = (eigensegment.unsqueeze(0) * features).mean(dim=(1,2))

        ############# Objects #############
        # Get segment
        object_segment = (eigenvector > threshold).float()
        if 0.5 < torch.mean(object_segment).item() < 1.0:  # reverse segment
            object_segment = (1 - object_segment)
        # Do CRF with unary potentials U
        U = F.interpolate(
            object_segment.reshape(1, 1, H_patch, W_patch), 
            size=(H_pad, W_pad), mode='bilinear', align_corners=False
        ).squeeze()
        U = torch.stack((1 - U, U), dim=-1)
        object_segment = denseCRF.densecrf(img_np, U.numpy(), crf_params)
        # In case there is no object
        if np.sum(np.abs(object_segment)) == 0:
            object_segment = 1 - object_segment
        # Get largest connected component
        object_segment = get_largest_cc(object_segment)
        object_segment = torch.from_numpy(object_segment)
        # Get and pool features
        object_pooled_feature = (object_segment.unsqueeze(0) * features).mean(dim=(1,2))

        ############# Output #############
        new_data_dict['eigenvectors'].append(eigenvector)
        new_data_dict['eigensegments'].append(eigensegment)
        new_data_dict['pooled_features'].append(pooled_feature)
        new_data_dict['eigensegments_object'].append(object_segment)
        new_data_dict['pooled_features_object'].append(object_pooled_feature)
    new_data_dict['eigenvectors'] = torch.stack(new_data_dict['eigenvectors'])
    new_data_dict['eigensegments'] = torch.stack(new_data_dict['eigensegments'])
    new_data_dict['pooled_features'] = torch.stack(new_data_dict['pooled_features'])
    new_data_dict['eigensegments_object'] = torch.stack(new_data_dict['eigensegments_object'])
    new_data_dict['pooled_features_object'] = torch.stack(new_data_dict['pooled_features_object'])
    new_data_dict = dict(new_data_dict)
    # Save dict
    output_file = str(Path(output_dir) / f'{prefix}-eigensegments-{index:05d}.pth')
    torch.save(new_data_dict, output_file)


@torch.no_grad()
def create_segments(
    prefix: str,
    features_root: str = './features',
    output_dir: str = './object_eigensegments',
    K: int = 3, 
    threshold: float = 0.0, 
    multiprocessing: int = 0
):
    """
    Example:
    python extract-segments.py create_segments \
        --prefix VOC2007-dino_vits16 \
        --features_root ./features \
        --output_dir ./object_eigensegments \
    """
    start = time.time()
    _create_object_segment_fn = partial(_create_object_segment, K=K, threshold=threshold, crf_params=CRF_PARAMS,
                                        prefix=prefix, output_dir=output_dir)
    inputs = list(enumerate(sorted(Path(features_root).iterdir())))  # inputs are (index, files) tuples
    if multiprocessing:
        from multiprocessing import Pool
        with Pool(multiprocessing) as pool:
            list(tqdm(pool.imap(_create_object_segment_fn, inputs), total=len(inputs)))
    else:
        for inp in tqdm(inputs):
            _create_object_segment_fn(inp)
    print(f'Done in {time.time() - start:.1f}s')


def combine_segments(
    features_root: str = './features',
    segments_root: str = './eigensegments',
    output_file: str = './tmp.pth',
):
    """
    Example:
    python extract-segments.py combine_segments \
            --features_root ./features \
            --segments_root ./eigensegments \
            --output_file ./eigensegments/VOC2007-dino_vits16-processed.pth
    """
    
    # Load
    feature_files = sorted(Path(features_root).iterdir())
    segment_files = sorted(Path(segments_root).iterdir())
    assert len(feature_files) == len(segment_files)

    # Combine
    combined_output_dict = defaultdict(list)
    for i, (ff, fs) in enumerate(tqdm(list(zip(feature_files, segment_files)))):
        features_dict = torch.load(ff, map_location='cpu')
        segments_dict = torch.load(fs, map_location='cpu')
        for k, v in features_dict.items():
            combined_output_dict[k].append(v)
        for k, v in segments_dict.items():
            combined_output_dict[k].append(v)
    combined_output_dict = dict(combined_output_dict)

    # Save
    torch.save(combined_output_dict, output_file)
    print(f'Saved file to {output_file}')
    

def vis_segments(
    input_file: str = './VOC2007-dino_vits16-processed.pth',
):
    """
    Example:
    streamlit run extract-segments.py vis_segments -- \
            --input_file ./VOC2007-dino_vits16-processed.pth
    """
    # Streamlit setup
    import streamlit as st
    st.set_page_config(layout='wide')

    # Load
    data_dict = torch.load(input_file, map_location='cpu')
    for k in data_dict:
        st.write(k, len(data_dict[k]), type(data_dict[k][0]))

    # Display
    column = 'object_eigensegments'  # 'eigensegments'
    for i in range(20):
        img = data_dict['images_resized'][i]
        seg = data_dict[column][i].numpy() * 255
        image = _inverse_transform(img.squeeze(0))
        cols = st.columns(6)
        cols[0].image(image)
        for j in range(5):
            cols[j + 1].image(seg[j])


if __name__ == '__main__':
    fire.Fire(dict(
        create_segments=create_segments, 
        combine_segments=combine_segments, 
        vis_segments=vis_segments
    ))
