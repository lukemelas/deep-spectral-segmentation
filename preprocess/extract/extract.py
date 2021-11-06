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
from sklearn.cluster import MiniBatchKMeans
try:
    from sklearnex.cluster import KMeans, DBSCAN
    print('Using sklearnex (accelerated sklearn)')
except:
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
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_multi_region_segmentation(
    inp: Tuple[int, Tuple[str, str]], 
    adaptive: bool, 
    adaptive_eigenvalue_threshold: float, 
    non_adaptive_num_segments: int,
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
    segmap = clusters.reshape(H_patch, W_patch)

    # TODO: Improve this step in the pipeline.
    # Background detection: we assume that the segment with the most border pixels is the 
    # background region. We will always make this region equal 0. 
    indices, normlized_counts = utils.get_border_fraction(segmap)
    bg_index = indices[np.argmax(normlized_counts)].item()
    bg_region = (segmap == bg_index)
    zero_region = (segmap == 0)
    segmap[bg_region] = 0
    segmap[zero_region] = bg_index

    # Save dict
    Image.fromarray(segmap).convert('L').save(output_file)


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
    python extract.py extract_multi_region_segmentation \
        --features_dir "./data/VOC2012/features" \
        --eigs_dir "./data/VOC2012/eigs" \
        --output_dir "./data/VOC2012/multi_region_segmentation" \
    """
    fn = partial(_extract_multi_region_segmentation, adaptive=adaptive, 
                 adaptive_eigenvalue_threshold=adaptive_eigenvalue_threshold, 
                 non_adaptive_num_segments=non_adaptive_num_segments, output_dir=output_dir)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir, desc='Creating segmentations')
    utils.parallel_process(inputs, fn, multiprocessing)


def _extract_bbox(
    inp: Tuple[str, str],
    num_erode: int,
    num_dilate: int,
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
        if segment_index > 0:  # skip 0, because 0 is the background
            
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
):
    """
    Note: There is no need for multiprocessing here, as it is more convenient to save 
    the entire output as a single JSON file. Example:
    python extract.py extract_bboxes \
        --features_dir "./data/VOC2012/features" \
        --segmentations_dir "./data/VOC2012/multi_region_segmentation" \
        --num_erode 2 --num_dilate 5 \
        --output_file "./data/VOC2012/multi_region_bboxes/bboxes_e2_d5.pth" \
    """
    fn = partial(_extract_bbox, num_erode=num_erode, num_dilate=num_dilate)
    inputs = utils.get_paired_input_files(features_dir, segmentations_dir, desc='Processing segmentations')
    # utils.parallel_process(inputs, fn, multiprocessing)  # <-- if you want multiprocessing, but it's not necessary
    all_outputs = [fn(inp) for inp in inputs]
    torch.save(all_outputs, output_file)
    print('Done')


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
            --segmentations_root "./data/VOC2012/multi_region_segmentation"
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

    # Load
    for i, (image_path, segmap_path) in enumerate(zip(image_paths, segmap_paths)):
        if i > 20: break
        image_id = image_path.stem
        
        # Streamlit
        cols = []
        
        # Load
        image = np.array(Image.open(image_path).convert('RGB'))
        segmap = np.array(Image.open(segmap_path))
        segmap_fullres = cv2.resize(segmap, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        
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
            colors=colors[segmap_label_indices], bg_label=0, alpha=1.0)
        image_segmap_overlay = label2rgb(label=segmap_fullres, image=image, 
            colors=colors[segmap_label_indices], bg_label=0, alpha=0.45)
        segmap_caption = dict(zip(segmap_label_indices.tolist(), (segmap_label_counts).tolist()))

        # Streamlit
        cols.append({'image': blank_segmap_overlay, 'caption': segmap_caption})
        cols.append({'image': image_segmap_overlay, 'caption': segmap_caption})

        # Display
        for d, col in zip(cols, st.columns(len(cols))):
            col.image(**d)


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
            --bbox_file "./data/VOC2012/multi_region_bboxes/bboxes_e2_d5.pth" \
            --output_file "./data/VOC2012/features" \
            --output_file "./data/VOC2012/multi_region_bboxes/bbox_features_e2_d5.pth" \
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
        for (xmin, ymin, xmax, ymax) in bboxes:
            image_crop = image[:, :, ymin:ymax, xmin:xmax]
            feature_crop = model(image_crop)
            bbox_dict['feature'] = feature_crop.squeeze().cpu()
    
    # Save
    torch.save(bbox_list, output_file)
    print(f'Saved features to {output_file}')



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        extract_features=extract_features,  # step 1
        extract_eigs=extract_eigs,  # step 2
        extract_multi_region_segmentation=extract_multi_region_segmentation,  # step 3
        extract_bboxes=extract_bboxes,  # step 4
        extract_bbox_features=extract_bbox_features,  # step 5
        # extract_bbox_clusters=extract_bbox_clusters,  # step 6
        vis_segmentations=vis_segmentations,
    ))

