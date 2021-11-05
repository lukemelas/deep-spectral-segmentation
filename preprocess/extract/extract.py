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
from torchvision import transforms
from tqdm import tqdm
from typing import Callable, Iterable, List, Optional, Tuple, Union
from typing import Optional
import denseCRF
import fire
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
try:
    from sklearnex.cluster import KMeans, DBSCAN
    print('Using sklearnex (accelerated sklearn)')
except:
    from sklearn.cluster import KMeans, DBSCAN

import extract_utils as utils


@torch.no_grad()
def extract_features(
    images_list: str,
    images_root: Optional[str] = None,
    model_name: str = 'dino_vits16',
    batch_size: int = 1024,
    output_dir: str = './features',
):
    """
    Example:
        python extract.py extract_features \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --output_dir "./data/VOC2012/features" \
            --model_name dino_vits16 \
            --batch_size 1
    """

    # Models
    model_name_lower = model_name.lower()
    model, val_transform, patch_size, num_heads = utils.get_model(model_name_lower)

    # Add hook
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Dataset
    filenames = Path(images_list).read_text().splitlines()
    dataset = utils.ImagesDataset(filenames=filenames, images_root=images_root, transform=val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    print(f'Dataset size: {len(dataset)=}')
    print(f'Dataloader size: {len(dataloader)=}')

    # Prepare
    accelerator = Accelerator(fp16=True, cpu=False)
    model, dataloader = accelerator.prepare(model, dataloader)

    # Process
    for i, (images, files, indices) in enumerate(tqdm(dataloader, desc='Processing')):
        output_dict = {}

        # Reshape image
        P = patch_size
        B, C, H, W = images.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        # images = F.interpolate(images, size=(H_pad, W_pad), mode='bilinear')  # resize image
        images = images[:, :, :H_pad, :W_pad]

        # Forward
        out = accelerator.unwrap_model(model).get_intermediate_layers(images)[0].squeeze(0)

        # Reshape
        output_dict['out'] = out
        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        output_dict['k'] = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['q'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['v'] = output_qkv[2].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['indices'] = indices[0]
        output_dict['file'] = files[0]
        output_dict['id'] = id = Path(files[0]).stem
        output_dict['model_name'] = model_name
        output_dict['patch_size'] = 16
        output_dict['shape'] = (B, C, H, W)
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}

        # Save
        output_file = str(Path(output_dir) / f'{id}.pth')
        accelerator.save(output_dict, output_file)
        accelerator.wait_for_everyone()
    
    print(f'Saved features to {output_dir}')


def _extract_eig(
    inp: Tuple[int, str], 
    K: int, 
    images_root: str,
    output_dir: str,
    which_matrix: str = 'laplacian'
):
    index, features_file = inp

    # Load 
    data_dict = torch.load(features_file, map_location='cpu')
    image_id = data_dict['file'][:-4]
    
    # Load
    output_file = str(Path(output_dir) / f'{image_id}.pth')
    if Path(output_file).is_file():
        return  # skip because already generated

    # Load affinity matrix
    k_feats = data_dict['k'].squeeze()
    A = k_feats @ k_feats.T

    # Eigenvectors of affinity matrix
    if which_matrix == 'affinity_torch':
        eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
    
    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity':
        A = A.cpu().numpy()
        eigenvalues, eigenvectors = eigsh(A, which='LM', k=K)
        eigenvectors = torch.flip(torch.from_numpy(eigenvectors), dims=(-1,))
    
    # Eigenvectors of laplacian matrix
    elif which_matrix == 'laplacian':
        A = A.cpu().numpy()
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

    # Sign ambiguity
    for k in range(eigenvectors.shape[1]):
        if 0.5 < torch.mean((eigenvectors[:, k]).float()).item() < 1.0:  # reverse segment
            eigenvectors[:, k] = 0 - eigenvectors[:, k]

    # Save dict
    output_dict = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}
    torch.save(output_dict, output_file)


def extract_eigs(
    images_root: str,
    features_dir: str,
    output_dir: str,
    K: int = 5, 
    threshold: float = 0.0, 
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_eigs \
        --images_root "./data/VOC2012/images" \
        --features_dir "./data/VOC2012/features" \
        --output_dir "./data/VOC2012/eigs" \
    """
    fn = partial(_extract_eig, K=K, images_root=images_root, output_dir=output_dir)
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))  # inputs are (index, files) tuples
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_multi_region_segmentation(
    inp: Tuple[int, Tuple[str, str]], 
    adaptive: bool, 
    adaptive_eigenvalue_threshold: float, 
    non_adaptive_num_segments: int,
):
    index, (feature_path, eigs_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['file']).stem
    output_file = str(Path(output_dir) / f'{id}.png')
    if Path(output_file).is_file():
        return  # skip because already generated

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    
    # If adaptive, we use adaptive_eigenvalue_threshold and the eigenvalues to get an
    # adaptive number of segments per image. If not, we use non_adaptive_num_segments 
    # to get a fixed number of segments per image.
    if adaptive: 
        raise NotImplementedError()
    else:
        n_clusters = non_adaptive_num_segments

    # Eigenvector
    eigenvectors = data_dict['eigenvectors'][1:].numpy()  # take non-constant eigenvectors
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(eigenvectors.T)
    clusters = clusters.reshape(H_patch, W_patch)

    # Save dict
    Image.fromarray(clusters).convert('L').save(output_file)


def extract_multi_region_segmentation(
    features_dir: str,
    eigs_dir: str,
    output_dir: str,
    adaptive: bool = False,
    adaptive_eigenvalue_threshold: float = 0.9,
    non_adaptive_num_segments: int = 4,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_single_region_segmentation \
        --features_dir "./data/VOC2012/features" \
        --eigs_dir "./data/VOC2012/eigs" \

    """
    fn = partial(_extract_multi_region_segmentation, adaptive=adaptive, 
                 adaptive_eigenvalue_threshold=adaptive_eigenvalue_threshold, 
                 non_adaptive_num_segments=non_adaptive_num_segments, output_dir=output_dir)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir)
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_multilabel_mask(
    inp: Tuple[int, Tuple[str, str]], 
    adaptive: bool, 
    adaptive_eigenvalue_threshold: float, 
    non_adaptive_num_segments: int,
    prefix: str, output_dir: str,
    patch_size: int = 16
):
    index, (feature_path, segment_path) = inp
    # try:
    
    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(segment_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['file']).stem
    output_file = str(Path(output_dir) / f'{prefix}-mask-{id}.png')
    if Path(output_file).is_file():
        return  # skip because already generated

    # Sizes
    H_patch, W_patch = data_dict['shape'][-2] // patch_size, data_dict['shape'][-1] // patch_size 
    H_pad, W_pad = H_patch * patch_size, W_patch * patch_size

    if adaptive: 
        # use adaptive_eigenvalue_threshold and the eigenvalues to get an
        # adaptive number of segments per image
        raise NotImplementedError()
    else:
        # use non_adaptive_num_segments to get a fixed number of segments per image
        n_clusters = non_adaptive_num_segments

    # Eigenvector
    eigenvectors = data_dict['eigenvectors'][1:].numpy()  # take non-constant eigenvectors
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(eigenvectors.T)
    clusters = clusters.reshape(H_patch, W_patch)

    # Save dict
    Image.fromarray(clusters).convert('L').save(output_file)

    # except:
    #     print(f'Problem with index: {index}')
    #     return


def extract_multilabel_masks(
    prefix: str,
    features_root: str = './features_VOC2012',
    segments_root: str = './eigensegments_VOC2012',
    output_dir: str = './multilabel_masks_VOC2012',
    adaptive: bool = False,
    adaptive_eigenvalue_threshold: float = 0.9,
    non_adaptive_num_segments: int = 4,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_multilabel_masks \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --segments_root ./eigensegments_VOC2012 \
        --output_dir ./multilabel_masks_VOC2012 \
    """
    fn = partial(_extract_multilabel_mask, adaptive=adaptive, adaptive_eigenvalue_threshold=adaptive_eigenvalue_threshold, 
                 non_adaptive_num_segments=non_adaptive_num_segments, prefix=prefix, output_dir=output_dir)
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
    _parallel_process(inputs, fn, multiprocessing)


def extract_semantic_segmentations(
    prefix: str,
    features_root: str = './multilabel_masks_VOC2012',
    multilabel_masks_root: str = './multilabel_masks_VOC2012',
    output_dir: str = './semantic_segmentations_VOC2012',
    num_clusters_excluding_background: int = 20,
    use_background_heuristic: bool = True,
    border_background_heuristic_threshold: float = 0.60,
    roundness_background_heuristic_threshold: float = 0.05,
):
    """
    Example:
    python extract.py extract_semantic_segmentations \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --multilabel_masks_root ./multilabel_masks_VOC2012 \
        --output_dir ./semantic_segmentations_VOC2012 \
    """
    
    # Load
    feature_files = []
    multilabel_mask_files = []
    missing_files = 0
    pbar = tqdm(list(enumerate(sorted(Path(features_root).iterdir()))), desc=f'Loading features ({missing_files} missing)')
    for i, features_file in pbar:
        segmap_file = Path(multilabel_masks_root) / features_file.name.replace('features', 'mask').replace('.pth', '.png')
        if segmap_file.is_file():
            feature_files.append(str(features_file))
            multilabel_mask_files.append(str(segmap_file))
        else:
            missing_files += 1
            pbar.set_description(f'Loading features ({missing_files} missing)')
    print(f'Loaded {len(feature_files)} files. There were {missing_files} missing files.' )
    
    # Extract features
    segmaps = []  # (N, )
    image_ids = []  # (N, )
    bg_indices = []  # (N, )
    output_files = []  # (N, )
    all_segment_indices = []  # list of length N of arrays of size (num_clusters, )
    all_pooled_features = []  # list of length N of arrays of size (num_clusters, D)
    # <start> just trying out some other features  ---  TODO: REMOVE
    all_pooled_k_features = []
    all_pooled_v_features = []
    all_pooled_cls_features = []
    # <end>
    pbar = zip(tqdm(feature_files, desc='Loading and pooling features'), multilabel_mask_files)
    for features_file, segmap_file in pbar:
        data_dict = torch.load(str(features_file))
        segmap = np.array(Image.open(segmap_file))
        image_id = data_dict['id']

        # Output file
        output_file = str(Path(output_dir) / f'{prefix}-mask-{image_id}.png')
        # if Path(output_file).is_file():
        #     continue  # skip because already generated

        # Find the background index, if one exists
        bg_index = None
        if use_background_heuristic:
            maybe_bg_index = get_border_background_heuristic(segmap, border_background_heuristic_threshold)
            if maybe_bg_index is not None:
                if get_roundness_background_heuristic((segmap == bg_index), roundness_background_heuristic_threshold):
                    bg_index = maybe_bg_index

        # Resize features
        H_pad, W_pad = data_dict['images_resized'].shape[-2:]
        H_patch, W_patch = H_pad // 16, W_pad // 16
        assert (len(data_dict['out'][0].shape) == 3) and (data_dict['out'][0].shape[0] == 1) and (data_dict['out'][0].shape[1] == H_patch * W_patch + 1)
        features = data_dict['out'][0][0, 1:].reshape(H_patch, W_patch, data_dict['out'][0].shape[-1])  # (H_patch, W_patch, D)
        assert features.shape[:2] == segmap.shape
        segmap = segmap[..., None]  # (H_patch, H_patch, 1)

        # <start> just trying out some other features  ---  TODO: REMOVE
        k_features = data_dict['k'].reshape(H_patch, W_patch, data_dict['k'].shape[-1])  # (H_patch, W_patch, D)
        v_features = data_dict['v'].reshape(H_patch, W_patch, data_dict['v'].shape[-1])  # (H_patch, W_patch, D)
        cls_features = data_dict['out'][0][0, 0].reshape(data_dict['v'].shape[-1])  # (D, )
        image_pooled_k_features = []
        image_pooled_v_features = []
        image_pooled_cls_features = []
        # <end>

        # Loop over features
        image_segment_indices = []  # (num_clusters,)
        image_pooled_features = []  # (num_clusters, D)
        for index in np.unique(segmap):
            if index != bg_index:
                binary_mask = erode_or_dilate_mask(segmap == index, r=30, erode=False)  # sanity check
                segment_pooled_features = torch.mean(features * binary_mask, dim=(0, 1))
                image_segment_indices.append(index)
                image_pooled_features.append(segment_pooled_features)  # (D, )
                
                # <start> just trying out some other features  ---  TODO: REMOVE
                segment_pooled_k_features = torch.mean(k_features * binary_mask, dim=(0, 1))
                segment_pooled_v_features = torch.mean(v_features * binary_mask, dim=(0, 1))
                segment_pooled_v_features = torch.mean(v_features * binary_mask, dim=(0, 1))
                image_pooled_k_features.append(segment_pooled_k_features)  # (D, )
                image_pooled_v_features.append(segment_pooled_v_features)  # (D, )
                image_pooled_cls_features.append(cls_features)  # (D, )
                # <end>
            
        # Append
        segmaps.append(segmap)
        image_ids.append(image_id)
        bg_indices.append(bg_index)
        output_files.append(output_file)
        all_pooled_features.append(image_pooled_features)
        all_segment_indices.append(image_segment_indices)

        # <start> just trying out some other features  ---  TODO: REMOVE
        all_pooled_k_features.append(image_pooled_k_features)
        all_pooled_v_features.append(image_pooled_v_features)
        all_pooled_cls_features.append(image_pooled_cls_features)
        # <end>
        
    # Stack
    all_pooled_features_flat = torch.stack([feat for image_pooled_features in all_pooled_features for feat in image_pooled_features], dim=0).numpy()

    # <start> just trying out some other features  ---  TODO: REMOVE
    all_pooled_k_features_flat = torch.stack([feat for image_pooled_k_features in all_pooled_k_features for feat in image_pooled_k_features], dim=0).numpy()
    all_pooled_v_features_flat = torch.stack([feat for image_pooled_v_features in all_pooled_v_features for feat in image_pooled_v_features], dim=0).numpy()
    all_pooled_cls_features_flat = torch.stack([feat for image_pooled_cls_features in all_pooled_cls_features for feat in image_pooled_cls_features], dim=0).numpy()
    # <end>

    torch.save(dict(
        segmaps=segmaps, image_ids=image_ids, bg_indices=bg_indices, all_segment_indices=all_segment_indices, 
        all_pooled_features_flat=all_pooled_k_features_flat,
        all_pooled_k_features_flat=all_pooled_k_features_flat, 
        all_pooled_v_features_flat=all_pooled_v_features_flat,
        all_pooled_cls_features_flat=all_pooled_cls_features_flat,
    ), 'tmp/tmp_features.pth')
    return
    
    # # PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=32, whiten=True)
    # all_pooled_features_flat = pca.fit_transform(all_pooled_features_flat)

    # # Kmeans
    # n_clusters = num_clusters_excluding_background + (0 if use_background_heuristic else 1)
    # kmeans = MiniBatchKMeans(n_clusters=n_clusters)  # KMeans(n_clusters=n_clusters)
    # print(f'Starting KMeans. Features shape: {all_pooled_features_flat.shape}')
    # clusters = kmeans.fit_predict(all_pooled_features_flat)
    # if use_background_heuristic:
    #     clusters = clusters + 1 # increment all cluster indices because the background is 0
    # print(f'Completed KMeans. Clusters shape: {clusters.shape}')
    # print(f'Cluster counts: {np.unique(clusters, return_counts=True)[1]}')

    # Kmeans
    n_clusters = num_clusters_excluding_background + (0 if use_background_heuristic else 1)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)  # KMeans(n_clusters=n_clusters)
    print(f'Starting KMeans. Features shape: {all_pooled_features_flat.shape}')
    clusters = kmeans.fit_predict(all_pooled_features_flat)
    if use_background_heuristic:
        clusters = clusters + 1 # increment all cluster indices because the background is 0
    print(f'Completed KMeans. Clusters shape: {clusters.shape}')
    print(f'Cluster counts: {np.unique(clusters, return_counts=True)[1]}')

    # Save new segments
    index = 0  # to keep track of where we are in the flat clusters array
    pbar = enumerate(zip(tqdm(segmaps, desc='Mapping and saving segments'), image_ids, bg_indices, output_files, all_segment_indices))
    for i, (segmap, image_id, bg_index, output_file, image_segment_indices) in pbar:

        # Load the correct clusters
        image_clusters = clusters[index: index + len(image_segment_indices)]
        index += len(image_segment_indices)

        # Construct the semantic map, a map that goes from segment 
        # index (image-level) to cluster index (dataset-level)
        image_semantic_map = {}
        for segment_index, cluster_index in zip(image_segment_indices, image_clusters):
            image_semantic_map[segment_index] = cluster_index
        if bg_index is not None:
            image_semantic_map[bg_index] = 0

        # Reshape and check shapes
        segmap = segmap.squeeze(-1)
        _segmap_indices = set(np.unique(segmap).tolist())
        _image_semantic_map_indices = set(list(image_semantic_map.keys()))
        if not (_segmap_indices == _image_semantic_map_indices):
            import pdb
            pdb.set_trace()
        assert _segmap_indices == _image_semantic_map_indices, f'{_segmap_indices=} != {_image_semantic_map_indices=}'
        assert len(segmap.shape) == 2  # (H_patch, W_patch)
        assert cluster_index.min() >= 0  # just a check
        assert cluster_index.max() <= 255  # you'll have to save this as a non-grayscale uint8 image 

        # Apply the semantic map
        semantic_segmap = np.vectorize(image_semantic_map.get)(segmap)

        # Save output image
        Image.fromarray(semantic_segmap.astype(np.uint8)).convert('L').save(output_file)
    
    # Check
    assert index == len(clusters), f'{index=} but {len(clusters)=}'
    print('Done')


def _extract_crf_semantic_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    crf_params: Tuple, num_clusters: int,
    prefix: str, output_dir: str,
    patch_size: int = 16
):
    index, (feature_path, semantic_segmap_file) = inp
    # try:
    
    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    semantic_segmap = np.array(Image.open(semantic_segmap_file))

    # Sizes
    H, W = data_dict['shape'][-2], data_dict['shape'][-1] 
    H_patch, W_patch = H // patch_size, W // patch_size 
    H_pad, W_pad = H_patch * patch_size, W_patch * patch_size

    # Output file
    id = Path(data_dict['file']).stem
    output_file = str(Path(output_dir) / f'{prefix}-mask-{id}.png')
    # if Path(output_file).is_file():
    #     return  # skip because already generated

    # Load image
    image = _inverse_transform(data_dict['images_resized'].squeeze(0))
    img_np = np.array(image)
    
    # CRF
    semantic_segmap = torch.from_numpy(semantic_segmap).reshape(1, 1, H_patch, W_patch).to(torch.uint8)
    U = F.interpolate(semantic_segmap, size=(H_pad, W_pad), mode='nearest').squeeze()
    U = F.one_hot(U.long(), num_classes=num_clusters)
    semantic_segmap = denseCRF.densecrf(img_np, U, crf_params)

    # Eigenvector
    resized_semantic_segmap = resize(semantic_segmap, output_shape=(H, W))
    resized_semantic_segmap[:H_pad, :W_pad] = semantic_segmap

    # Save dict
    Image.fromarray(resized_semantic_segmap).convert('L').save(output_file)

    # except:
    #     print(f'Problem with index: {index}')
    #     return


def extract_crf_semantic_segmentations(
    prefix: str, 
    features_root: str = './features_VOC2012',
    semantic_segmentations_root: str = './semantic_segmentations_VOC2012',
    output_dir: str = './crf_semantic_segmentations_VOC2012',
    num_clusters: int = 21,
    multiprocessing: int = 0,
    # below: CRF parameters
    w1    = 20,     # weight of bilateral term  # 10.0,
    alpha = 30,    # spatial std  # 80,  
    beta  = 15,    # rgb  std  # 13,  
    w2    = 10,     # weight of spatial term  # 3.0, 
    gamma = 5,     # spatial std  # 3,   
    it    = 4.0,   # iteration  # 5.0, 
):
    """
    Example:
    python extract.py extract_crf_semantic_segmentations \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --semantic_segmentations_root ./semantic_segmentations_VOC2012 \
        --output_dir ./crf_semantic_segmentations_VOC2012 \
    """
    crf_params = (w1, alpha, beta, w2, gamma, it)
    fn = partial(_extract_crf_semantic_segmentations, num_clusters=num_clusters, crf_params=crf_params, prefix=prefix, output_dir=output_dir)
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=semantic_segmentations_root, segments_name='mask', ext='.png')
    _parallel_process(inputs, fn, multiprocessing)


def vis_segments(
    features_root: str = './features_VOC2012',
    segments_root: str = './eigensegments_VOC2012',
):
    """
    Example:
    streamlit run extract-segments.py vis_segments -- \
        --features_root ./features_VOC2012 \
        --segments_root ./eigensegments_VOC2012 \
    """
    # Streamlit setup
    import streamlit as st
    st.set_page_config(layout='wide')

    # Load
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
    print(f'{len(inputs)=}')

    # Combine
    for i, (features_file, fs) in inputs:
        if i > 20: break

        # Combine
        features_dict = torch.load(features_file, map_location='cpu')
        segments_dict = torch.load(fs, map_location='cpu')
        data_dict = defaultdict(list)
        for k, v in features_dict.items():
            data_dict[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
        for k, v in segments_dict.items():
            data_dict[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
        data_dict = dict(data_dict)

        # Print stuff
        if i == 0:
            for k, v in data_dict.items():
                st.write(k, type(v), v.shape if torch.is_tensor(v) else (v if isinstance(v, str) else None))

        # Display stuff
        img = data_dict['images_resized']
        image = _inverse_transform(img.squeeze(0))
        eig_seg = data_dict['eigensegments'].numpy() * 255
        obj_seg = data_dict['eigensegments_object'].numpy() * 255
        cols = st.columns(1 + 3 + 3)
        cols[0].image(image, caption=f'{data_dict["files"][0]} ({i})')
        cols[1].image(obj_seg[0], caption='obj seg 0')
        cols[2].image(obj_seg[1], caption='obj seg 1')
        cols[3].image(obj_seg[2], caption='obj seg 2')
        cols[4].image(eig_seg[0], caption='eig seg 0')
        cols[5].image(eig_seg[1], caption='eig seg 1')
        cols[6].image(eig_seg[2], caption='eig seg 2')


def vis_semantic_segmentations(
    images_file: str = './image-lists/VOC2012.txt',
    images_root: str = '/path/to/JPEGImages',
    semantic_segmentations_str: str = './semantic_segmentations_VOC2012/VOC2012-dino_vits16-mask-{image_id}.png',
):
    """
    Example:
    streamlit run extract-segments.py vis_semantic_segmentations -- \
        --images_file ./image-lists/VOC2012.txt \
        --images_root /data_q1_d/machine-learning-datasets/semantic-segmentation/PASCAL_VOC/VOC2012/VOCdevkit/VOC2012/JPEGImages \
        --semantic_segmentations_str ./crf_semantic_segmentations_VOC2012/VOC2012-dino_vits16-mask-{image_id}.png \
    """
    # Streamlit setup
    from skimage.color import label2rgb
    from matplotlib.cm import get_cmap
    import streamlit as st
    st.set_page_config(layout='wide')

    # Load
    colors = get_cmap('tab10', 21).colors[:, :3]
    for i, image_name in enumerate(Path(images_file).read_text().splitlines()):
        image_id = Path(image_name).stem
        semantic_segmap_file = semantic_segmentations_str.format(image_id=image_id)
        image_file = Path(images_root) / image_name
        # target_file = Path(image_root) / '..' / ''

        # Load
        image = np.array(Image.open(image_file).convert('RGB'))
        semantic_segmap = np.array(Image.open(semantic_segmap_file).resize(image.shape[:2][::-1], resample=Image.NEAREST))
        # semantic_segmap = resize(semantic_segmap, output_shape=, order=0).astype(semantic_segmap.dtype)

        # Color
        print(np.unique(semantic_segmap))
        blank_segmap_overlay = label2rgb(label=semantic_segmap, image=np.full_like(image, 255), 
            colors=colors[np.unique(semantic_segmap)], bg_label=0, alpha=1.0)
        image_segmap_overlay = label2rgb(label=semantic_segmap, image=image, 
            colors=colors[np.unique(semantic_segmap)], bg_label=0, alpha=0.45)

        # Display stuff
        cols = st.columns(5)
        cols[0].image(image, caption=image_id)
        # cols[1].image(blank_target_overlay, caption='target')
        # cols[2].image(image_target_overlay, caption='target')
        cols[1].image(blank_segmap_overlay, caption=str(np.unique(semantic_segmap).tolist()))
        cols[2].image(image_segmap_overlay, caption=str(np.unique(semantic_segmap).tolist()))

        if i > 60:
            break


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        extract_features=extract_features,
        extract_eigs=extract_eigs, 
        extract_object_mattes=extract_object_mattes,
        extract_object_masks=extract_object_masks,
        extract_multilabel_masks=extract_multilabel_masks,
        extract_semantic_segmentations=extract_semantic_segmentations,
        extract_crf_semantic_segmentations=extract_crf_semantic_segmentations,
        vis_segments=vis_segments,
        vis_semantic_segmentations=vis_semantic_segmentations,
    ))

