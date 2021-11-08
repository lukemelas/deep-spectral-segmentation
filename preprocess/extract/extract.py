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


def extract_features(
    images_list: str,
    images_root: Optional[str],
    model_name: str,
    batch_size: int,
    output_dir: str,
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
        output_dict['q'] = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
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
    # if Path(output_file).is_file():
    #     return  # skip because already generated

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
    which_matrix: str = 'laplacian',
    K: int = 5,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_eigs \
        --images_root "./data/VOC2012/images" \
        --features_dir "./data/VOC2012/features" \
        --which_matrix "laplacian" \
        --output_dir "./data/VOC2012/eigs/laplacian" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_eig, K=K, which_matrix=which_matrix, images_root=images_root, output_dir=output_dir)
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_multi_region_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    adaptive: bool, 
    non_adaptive_num_segments: int,
    infer_bg_index: bool,
    output_dir: str,
):
    index, (feature_path, eigs_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['id'])
    output_file = str(Path(output_dir) / f'{id}.png')
    # if Path(output_file).is_file():
    #     return  # skip because already generated

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    
    # If adaptive, we use the gaps between eigenvalues to determine the number of 
    # segments per image. If not, we use non_adaptive_num_segments to get a fixed
    # number of segments per image.
    if adaptive:
        indices_by_gap = np.argsort(np.diff(data_dict['eigenvalues'].numpy()))[::-1]
        index_largest_gap = indices_by_gap[indices_by_gap != 0][0]  # remove zero and take the biggest
        n_clusters = index_largest_gap + 1
        # print(f'Number of clusters: {n_clusters}')
    else:
        n_clusters = non_adaptive_num_segments

    # Eigenvector
    eigenvectors = data_dict['eigenvectors'][1:].numpy()  # take non-constant eigenvectors
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(eigenvectors.T)
    segmap = clusters.reshape(H_patch, W_patch)

    # TODO: Improve this step in the pipeline.
    # Background detection: we assume that the segment with the most border pixels is the 
    # background region. We will always make this region equal 0. 
    if infer_bg_index:
        indices, normlized_counts = utils.get_border_fraction(segmap)
        bg_index = indices[np.argmax(normlized_counts)].item()
        bg_region = (segmap == bg_index)
        zero_region = (segmap == 0)
        segmap[bg_region] = 0
        segmap[zero_region] = bg_index

    # Save dict
    Image.fromarray(segmap).convert('L').save(output_file)


def extract_multi_region_segmentations(
    features_dir: str,
    eigs_dir: str,
    output_dir: str,
    adaptive: bool = False,
    non_adaptive_num_segments: int = 4,
    infer_bg_index: bool = True,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_multi_region_segmentations \
        --features_dir "./data/VOC2012/features" \
        --eigs_dir "./data/VOC2012/eigs/laplacian" \
        --output_dir "./data/VOC2012/multi_region_segmentation/fixed" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_multi_region_segmentations, adaptive=adaptive, infer_bg_index=infer_bg_index,
                 non_adaptive_num_segments=non_adaptive_num_segments, output_dir=output_dir)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir)
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_single_region_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    threshold: float,
    output_dir: str,
):
    index, (feature_path, eigs_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['id'])
    output_file = str(Path(output_dir) / f'{id}.png')
    if Path(output_file).is_file():
        return  # skip because already generated

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    
    # Eigenvector
    eigenvector = data_dict['eigenvectors'][1].numpy()  # take smallest non-zero eigenvector
    segmap = (eigenvector > threshold).reshape(H_patch, W_patch)

    # Save dict
    Image.fromarray(segmap).convert('L').save(output_file)


def extract_single_region_segmentations(
    features_dir: str,
    eigs_dir: str,
    output_dir: str,
    threshold: float = 0.0,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_single_region_segmentations \
        --features_dir "./data/VOC2012/features" \
        --eigs_dir "./data/VOC2012/eigs/laplacian" \
        --output_dir "./data/VOC2012/single_region_segmentation/patches" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_single_region_segmentations, threshold=threshold, output_dir=output_dir)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir)
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_bbox(
    inp: Tuple[str, str],
    num_erode: int,
    num_dilate: int,
    skip_bg_index: bool,
):
    index, (feature_path, segmentation_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    segmap = np.array(Image.open(str(segmentation_path)))
    image_id = data_dict['id']

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)

    # Get bounding boxes
    outputs = {'bboxes': [], 'bboxes_original_resolution': [], 'segment_indices': [], 'id': image_id, 
               'format': "(xmin, ymin, xmax, ymax)"}
    for segment_index in sorted(np.unique(segmap).tolist()):
        if (not skip_bg_index) or (segment_index > 0):  # skip 0, because 0 is the background
            
            # Erode and dilate mask
            binary_mask = (segmap == segment_index)
            binary_mask = utils.erode_or_dilate_mask(binary_mask, r=num_erode, erode=True)
            binary_mask = utils.erode_or_dilate_mask(binary_mask, r=num_dilate, erode=False)

            # Find box
            mask = np.where(binary_mask == 1)
            ymin, ymax = min(mask[0]), max(mask[0]) + 1  # add +1 because excluded max
            xmin, xmax = min(mask[1]), max(mask[1]) + 1  # add +1 because excluded max
            bbox = [xmin, ymin, xmax, ymax]
            bbox_resized = [x * P for x in bbox]  # rescale to image size
            bbox_features = [ymin, xmin, ymax, xmax]  # feature space coordinates are different

            # Append
            outputs['segment_indices'].append(segment_index)
            outputs['bboxes'].append(bbox)
            outputs['bboxes_original_resolution'].append(bbox_resized)
    
    return outputs


def extract_bboxes(
    features_dir: str,
    segmentations_dir: str,
    output_file: str,
    num_erode: int = 2,
    num_dilate: int = 3,
    skip_bg_index: bool = True,
):
    """
    Note: There is no need for multiprocessing here, as it is more convenient to save 
    the entire output as a single JSON file. Example:
    python extract.py extract_bboxes \
        --features_dir "./data/VOC2012/features" \
        --segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed" \
        --num_erode 2 --num_dilate 5 \
        --output_file "./data/VOC2012/multi_region_bboxes/fixed/bboxes_e2_d5.pth" \
    """
    utils.make_output_dir(str(Path(output_file).parent), check_if_empty=False)
    fn = partial(_extract_bbox, num_erode=num_erode, num_dilate=num_dilate, skip_bg_index=skip_bg_index)
    inputs = utils.get_paired_input_files(features_dir, segmentations_dir)
    all_outputs = [fn(inp) for inp in tqdm(inputs, desc='Extracting bounding boxes')]
    torch.save(all_outputs, output_file)
    print('Done')


def extract_bbox_features(
    images_root: str,
    bbox_file: str,
    model_name: str,
    output_file: str,
):
    """
    Example:
        python extract.py extract_bbox_features \
            --model_name dino_vits16 \
            --images_root "./data/VOC2012/images" \
            --bbox_file "./data/VOC2012/multi_region_bboxes/fixed/bboxes_e2_d5.pth" \
            --output_file "./data/VOC2012/features" \
            --output_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_features_e2_d5.pth" \
    """

    # Load bounding boxes
    bbox_list = torch.load(bbox_file)
    total_num_boxes = sum(len(d['bboxes']) for d in bbox_list)
    print(f'Loaded bounding box list. There are {total_num_boxes} total bounding boxes.')

    # Models
    model_name_lower = model_name.lower()
    model, val_transform, patch_size, num_heads = utils.get_model(model_name_lower)
    model.eval().to('cuda')

    # Loop over boxes
    for bbox_dict in tqdm(bbox_list):
        # Get image info
        image_id = bbox_dict['id']
        bboxes = bbox_dict['bboxes_original_resolution']
        # Load image as tensor
        image_filename = str(Path(images_root) / f'{image_id}.jpg')
        image = val_transform(Image.open(image_filename).convert('RGB'))  # (3, H, W)
        image = image.unsqueeze(0).to('cuda')  # (1, 3, H, W)
        features_crops = []
        for (xmin, ymin, xmax, ymax) in bboxes:
            image_crop = image[:, :, ymin:ymax, xmin:xmax]
            features_crop = model(image_crop).squeeze().cpu()
            features_crops.append(features_crop)
        bbox_dict['features'] = torch.stack(features_crops, dim=0)
    
    # Save
    torch.save(bbox_list, output_file)
    print(f'Saved features to {output_file}')


def extract_bbox_clusters(
    bbox_features_file: str,
    output_file: str,
    num_clusters: int = 21, 
    seed: int = 0, 
    pca_dim: Optional[int] = 32,
):
    """
    Example:
        python extract.py extract_bbox_clusters \
            --bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_features_e2_d5.pth" \
            --pca_dim 32 --num_clusters 20 --seed 0 \
            --output_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_clusters_e2_d5_pca_32.pth" \
    """

    # Load bounding boxes
    bbox_list = torch.load(bbox_features_file)
    total_num_boxes = sum(len(d['bboxes']) for d in bbox_list)
    print(f'Loaded bounding box list. There are {total_num_boxes} total bounding boxes with features.')

    # Loop over boxes and stack features with PyTorch, because Numpy is too slow
    print(f'Stacking and normalizing features')
    all_features = torch.cat([bbox_dict['features'] for bbox_dict in bbox_list], dim=0)  # (numBbox, D)
    all_features = all_features / torch.norm(all_features, dim=-1, keepdim=True)  # (numBbox, D)f
    all_features = all_features.numpy()

    # Cluster: PCA
    if pca_dim:
        pca = PCA(pca_dim)
        print(f'Computing PCA with dimension {pca_dim}')
        all_features = pca.fit_transform(all_features)

    # Cluster: K-Means
    print(f'Computing K-Means clustering with {num_clusters} clusters')
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=4096, max_iter=5000, random_state=seed)
    clusters = kmeans.fit_predict(all_features)
    
    # Print 
    _indices, _counts = np.unique(clusters, return_counts=True)
    print(f'Cluster indices: {_indices.tolist()}')
    print(f'Cluster counts: {_counts.tolist()}')

    # Loop over boxes and add clusters
    idx = 0
    for bbox_dict in bbox_list:
        num_bboxes = len(bbox_dict['bboxes'])
        del bbox_dict['features']  # bbox_dict['features'] = bbox_dict['features'].squeeze()
        bbox_dict['clusters'] = clusters[idx: idx + num_bboxes]
        idx = idx + num_bboxes
    
    # Save
    torch.save(bbox_list, output_file)
    print(f'Saved features to {output_file}')


def extract_semantic_segmentations(
    segmentations_dir: str,
    bbox_clusters_file: str,
    output_dir: str,
):
    """
    Example:
        python extract.py extract_semantic_segmentations \
            --segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed" \
            --bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_clusters_e2_d5_pca_32.pth" \
            --output_dir "./data/VOC2012/semantic_segmentations/patches/fixed/segmaps_e2_d5_pca_32" \
    """

    # Load bounding boxes
    bbox_list = torch.load(bbox_clusters_file)
    total_num_boxes = sum(len(d['bboxes']) for d in bbox_list)
    print(f'Loaded bounding box list. There are {total_num_boxes} total bounding boxes with features and clusters.')

    # Output
    utils.make_output_dir(output_dir)

    # Loop over boxes
    for bbox_dict in tqdm(bbox_list):
        # Get image info
        image_id = bbox_dict['id']
        # Load segmentation as tensor
        segmap_path = str(Path(segmentations_dir) / f'{image_id}.png')
        segmap = np.array(Image.open(segmap_path))
        # Check if the segmap is a binary file with foreground pixels saved as 255 instead of 1
        # this will be the case for some of our baselines
        if set(np.unique(segmap).tolist()).issubset({0, 255}):
            segmap[segmap == 255] = 1  
        # Semantic map
        if not len(bbox_dict['segment_indices']) == len(bbox_dict['clusters'].tolist()):
            import pdb
            pdb.set_trace()
        semantic_map = dict(zip(bbox_dict['segment_indices'], bbox_dict['clusters'].tolist()))
        assert 0 not in semantic_map, semantic_map
        semantic_map[0] = 0  # background region remains zero
        # Perform mapping
        semantic_segmap = np.vectorize(semantic_map.__getitem__)(segmap)
        # Save
        output_file = str(Path(output_dir) / f'{image_id}.png')
        Image.fromarray(semantic_segmap.astype(np.uint8)).convert('L').save(output_file)
    
    print(f'Saved features to {output_dir}')


def _extract_crf_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    images_root: str,
    num_classes: int,
    output_dir: str,
    crf_params: Tuple,
):
    index, (image_file, segmap_path) = inp

    # Output file
    id = Path(image_file).stem
    output_file = str(Path(output_dir) / f'{id}.png')
    # if Path(output_file).is_file():
    #     return  # skip because already generated

    # Load image and segmap
    image_file = str(Path(images_root) / f'{id}.jpg')
    image = np.array(Image.open(image_file).convert('RGB'))  # (H_patch, W_patch, 3)
    segmap = np.array(Image.open(segmap_path))  # (H_patch, W_patch)
     
    # Sizes
    H, W = image.shape[:2]
    H_patch, W_patch = H // 16, W // 16
    H_pad, W_pad = H_patch * 16, W_patch * 16

    # Resize and expand
    segmap_upscaled = cv2.resize(segmap, dsize=(W_pad, H_pad), interpolation=cv2.INTER_NEAREST)  # (H_pad, W_pad)
    segmap_orig_res = cv2.resize(segmap, dsize=(W, H), interpolation=cv2.INTER_NEAREST)  # (H, W)
    segmap_orig_res[:H_pad, :W_pad] = segmap_upscaled  # replace with the correctly upscaled version, just in case they are different

    # CRF
    unary_potentials = F.one_hot(torch.from_numpy(segmap_orig_res).long(), num_classes=num_classes)
    segmap_crf = denseCRF.densecrf(image, unary_potentials, crf_params)  # (H_pad, W_pad)

    # Save
    Image.fromarray(segmap_crf).convert('L').save(output_file)


def extract_crf_segmentations(
    images_list: str,
    images_root: str,
    segmentations_dir: str,
    output_dir: str,
    num_classes: int = 21,
    multiprocessing: int = 0,
    # CRF parameters
    w1    = 20,    # weight of bilateral term  # 10.0,
    alpha = 30,    # spatial std  # 80,  
    beta  = 13,    # rgb  std  # 13,  
    w2    = 5,     # weight of spatial term  # 3.0, 
    gamma = 3,     # spatial std  # 3,   
    it    = 5.0,   # iteration  # 5.0, 
):
    """
    Example:
    python extract.py extract_crf_segmentations \
        --images_list "./data/VOC2012/lists/images.txt" \
        --images_root "./data/VOC2012/images" \
        --segmentations_dir "./data/VOC2012/semantic_segmentations/patches/fixed/segmaps_e2_d5_pca_32" \
        --output_dir "./data/VOC2012/semantic_segmentations/crf/fixed/segmaps_e2_d5_pca_32" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_crf_segmentations, images_root=images_root, num_classes=num_classes, output_dir=output_dir,
                 crf_params=(w1, alpha, beta, w2, gamma, it))
    inputs = utils.get_paired_input_files(images_list, segmentations_dir)
    print(f'Found {len(inputs)} images and segmaps')
    utils.parallel_process(inputs, fn, multiprocessing)


def vis_segmentations(
    images_list: str,
    images_root: str,
    segmentations_root: str,
    bbox_file: Optional[str] = None,
):
    """
    Example:
        streamlit run extract.py vis_segmentations -- \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --segmentations_root "./data/VOC2012/multi_region_segmentation/fixed"
    or alternatively:
            --segmentations_root "./data/VOC2012/semantic_segmentations/crf/fixed/segmaps_e2_d5_pca_32/"
    """
    # Streamlit setup
    from skimage.color import label2rgb
    from matplotlib.cm import get_cmap
    import streamlit as st
    st.set_page_config(layout='wide')

    # Inputs
    image_paths = []
    segmap_paths = []
    images_root = Path(images_root)
    segmentations_root = Path(segmentations_root)
    for image_file in Path(images_list).read_text().splitlines():
        segmap_file = f'{Path(image_file).stem}.png'
        image_paths.append(images_root / image_file)
        segmap_paths.append(segmentations_root / segmap_file)
    print(f'Found {len(image_paths)} image and segmap paths')

    # Load optional bounding boxes
    if bbox_file is not None:
        bboxes_list = torch.load(bbox_file)

    # Colors
    colors = get_cmap('tab20', 21).colors[:, :3]

    # Which index
    which_index = st.number_input(label='Which index to view (0 for all)', value=0)

    # Load
    total = 0
    for i, (image_path, segmap_path) in enumerate(zip(image_paths, segmap_paths)):
        if total > 40: break
        image_id = image_path.stem
        
        # Streamlit
        cols = []
        
        # Load
        image = np.array(Image.open(image_path).convert('RGB'))
        segmap = np.array(Image.open(segmap_path))
        segmap_fullres = cv2.resize(segmap, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Only view images with a specific class
        if which_index not in np.unique(segmap):
            continue
        total += 1

        # Streamlit
        cols.append({'image': image, 'caption': image_id})

        # Load optional bounding boxes
        bboxes = None
        if bbox_file is not None:
            bboxes = torch.tensor(bboxes_list[i]['bboxes_original_resolution'])
            assert bboxes_list[i]['id'] == image_id, f"{bboxes_list[i]['id']=} but {image_id=}"
            image_torch = torch.from_numpy(image).permute(2, 0, 1)
            image_with_boxes_torch = draw_bounding_boxes(image_torch, bboxes)
            image_with_boxes = image_with_boxes_torch.permute(1, 2, 0).numpy()
            
            # Streamlit
            cols.append({'image': image_with_boxes})
            
        # Color
        segmap_label_indices, segmap_label_counts = np.unique(segmap, return_counts=True)
        blank_segmap_overlay = label2rgb(label=segmap_fullres, image=np.full_like(image, 128), 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=1.0)
        image_segmap_overlay = label2rgb(label=segmap_fullres, image=image, 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=0.45)
        segmap_caption = dict(zip(segmap_label_indices.tolist(), (segmap_label_counts).tolist()))

        # Streamlit
        cols.append({'image': blank_segmap_overlay, 'caption': segmap_caption})
        cols.append({'image': image_segmap_overlay, 'caption': segmap_caption})

        # Display
        for d, col in zip(cols, st.columns(len(cols))):
            col.image(**d)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        extract_features=extract_features,  # step 1
        extract_eigs=extract_eigs,  # step 2
        # multi-region pipeline
        extract_multi_region_segmentations=extract_multi_region_segmentations,  # step 3
        extract_bboxes=extract_bboxes,  # step 4
        extract_bbox_features=extract_bbox_features,  # step 5
        extract_bbox_clusters=extract_bbox_clusters,  # step 6
        extract_semantic_segmentations=extract_semantic_segmentations,  # step 7
        extract_crf_segmentations=extract_crf_segmentations,  # step 8
        # single
        extract_single_region_segmentations=extract_single_region_segmentations, 
        # vis
        vis_segmentations=vis_segmentations,  # optional
    ))