"""An experimental script to create eigensegments"""
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union
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
from skimage.morphology import binary_erosion, binary_dilation
from skimage.transform import resize
from multiprocessing import Pool
from sklearn.cluster import KMeans
from skimage.measure import label as measure_label
from skimage.measure import perimeter as measure_perimeter


########################### HELPERS ###########################


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


def erode_or_dilate_mask(x: Union[torch.Tensor, np.ndarray], r: int = 0, erode=True):
    fn = binary_erosion if erode else binary_dilation
    for _ in range(r):
        x = fn(x)
    return x


def trimap_from_mask(mask, r=15):
    is_fg = erode_or_dilate_mask(mask, r=r, erode=True)
    is_bg = ~(erode_or_dilate_mask(mask, r=r, erode=False))
    if is_fg.sum() == 0:
        is_fg = erode_or_dilate_mask(mask, r=1, erode=True)
    trimap = np.full_like(mask, fill_value=0.5, dtype=float)
    trimap[is_fg] = 1.0
    trimap[is_bg] = 0.0
    return trimap


def get_roundness(mask: np.array):
    r"""Get roundness := (4 pi area) / perimeter^2"""

    # Get connected components
    component, num = measure_label(mask, return_num=True, background=0)
    if num == 0:
        return 0

    # Get area of biggest connected component
    areas, perimeters = [], []
    for i in range(1, num + 1):
        component_i = (component == i)
        area = np.sum(component_i)
        perimeter = measure_perimeter(component_i)
        areas.append(area)
        perimeters.append(perimeter)
    max_component = np.argmax(areas)
    max_component_area = areas[max_component]
    num_pixels = mask.shape[-1] * mask.shape[-2]
    if num_pixels * 0.05 < max_component_area < num_pixels * 0.90:
        max_component_perimeter = perimeters[max_component]
        roundness = max_component_area / max_component_perimeter ** 2
        return roundness
    else:
        return 0


def get_border_fraction(segmap: np.array):
    num_border_pixels = 2 * (segmap.shape[0] + segmap.shape[1])
    counts_map = {idx: 0 for idx in np.unique(segmap)}
    np.zeros(len(np.unique(segmap)))
    for border in [segmap[:, 0], segmap[:, -1], segmap[0, :], segmap[-1, :]]:
        unique, counts = np.unique(border, return_counts=True)
        for idx, count in zip(unique.tolist(), counts.tolist()):
            counts_map[idx] += count
    # normlized_counts_map = {idx: count / num_border_pixels for idx, count in counts_map.items()}
    indices = np.array(list(counts_map.keys()))
    normlized_counts = np.array(list(counts_map.values())) / num_border_pixels
    return indices, normlized_counts


def get_border_background_heuristic(segmap: np.array, threshold: float = 0.60) -> Optional[int]:
    indices, normlized_counts = get_border_fraction(segmap)
    if np.max(normlized_counts) > threshold:
        return indices[np.argmax(normlized_counts)].item()
    return None


def get_roundness_background_heuristic(mask: np.array, threshold: float = 0.05) -> bool:
    return get_roundness(mask) < threshold  # returns False if the background is too round


########################### SCRIPTS ###########################


def _get_feature_and_segment_inputs(features_root, segments_root, segments_name='eigensegments', ext='.pth') -> List[Tuple[int, Tuple[str, str]]]:
    inputs = []  # inputs are (index, (feature_file, segment_file)) tuples
    missing_files = 0
    for p in tqdm(sorted(Path(features_root).iterdir()), desc='Loading file list'):
        features_file = str(p)
        segments_file = str(Path(segments_root) / p.name.replace('features', segments_name).replace('.pth', ext))
        if Path(features_file).is_file() and Path(segments_file).is_file():
            inputs.append((features_file, segments_file))
        else:
            missing_files += 1
    print(f'Loaded {len(inputs)} files. There were {missing_files} missing files.' )
    inputs = list(enumerate(inputs))
    return inputs


def _parallel_process(inputs: Iterable, fn: Callable, multiprocessing: int = 0):
    start = time.time()
    if multiprocessing:
        print('Starting multiprocessing')
        with Pool(multiprocessing) as pool:
            for _ in tqdm(pool.imap(fn, inputs), total=len(inputs)):
                pass
    else:
        for inp in tqdm(inputs):
            fn(inp)
    print(f'Finished in {time.time() - start:.1f}s')


def _create_object_segment(
    inp: Tuple[int, str], K: int, threshold: float, crf_params: Tuple, 
    prefix: str, output_dir: str, patch_size: int = 16
):
    index, path = inp
    
    # try:
    if True:
    
        # Load 
        data_dict = torch.load(path, map_location='cpu')
        image_id = data_dict['file'][:-4]
        
        # Load
        output_file = str(Path(output_dir) / f'{prefix}-eigensegments-{image_id}.pth')
        if Path(output_file).is_file():
            return  # skip because already generated

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
        
        # # Eigenvectors of affinity matrix with scipy
        # from scipy.sparse.linalg import eigsh
        # A = k_feats @ k_feats.T
        # eigenvalues, eigenvectors = eigsh(A.cpu().numpy(), which='LM', k=K)  # find small eigenvalues
        # eigenvectors = torch.flip(torch.from_numpy(eigenvectors), dims=(-1,))
        
        # # Eigenvectors of laplacian matrix
        from scipy.sparse.linalg import eigsh
        A = (k_feats @ k_feats.T).cpu().numpy()
        _W_semantic = (A * (A > 0))
        _W_semantic = _W_semantic / _W_semantic.max()
        diag = _W_semantic @ np.ones(_W_semantic.shape[0])
        diag[diag < 1e-12] = 1.0
        D = np.diag(diag)  # row sum
        try:
            eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=K, sigma=0, which='LM', M=D)
        except:
            eigenvalues, eigenvectors = eigsh(D - _W_semantic, k=K, which='SM', M=D)
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors).float()

        # CRF
        new_data_dict = defaultdict(list)
        for k in range(K):
            eigenvalue = eigenvalues[k]
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
            new_data_dict['eigenvalues'].append(eigenvalue)
            new_data_dict['eigenvectors'].append(eigenvector)
            new_data_dict['eigensegments'].append(eigensegment)
            new_data_dict['pooled_features'].append(pooled_feature)
            new_data_dict['eigensegments_object'].append(object_segment)
            new_data_dict['pooled_features_object'].append(object_pooled_feature)
        new_data_dict['eigenvalues'] = torch.stack(new_data_dict['eigenvalues'])
        new_data_dict['eigenvectors'] = torch.stack(new_data_dict['eigenvectors'])
        new_data_dict['eigensegments'] = torch.stack(new_data_dict['eigensegments'])
        new_data_dict['pooled_features'] = torch.stack(new_data_dict['pooled_features'])
        new_data_dict['eigensegments_object'] = torch.stack(new_data_dict['eigensegments_object'])
        new_data_dict['pooled_features_object'] = torch.stack(new_data_dict['pooled_features_object'])
        new_data_dict['file'] = data_dict['file']
        new_data_dict['id'] = data_dict.get('id', image_id)
        new_data_dict = dict(new_data_dict)
        # Save dict
        torch.save(new_data_dict, output_file)

    # except Exception as e:
    #     if isinstance(e, KeyboardInterrupt):
    #         import sys
    #         sys.exit()
    #     print(f'Problem with {index=}')


def create_segments(
    prefix: str,
    features_root: str = './features',
    output_dir: str = './eigensegments',
    K: int = 5, 
    threshold: float = 0.0, 
    multiprocessing: int = 0
):
    """
    Example:
    python extract-segments.py create_segments \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features \
        --output_dir ./eigensegments \
    """
    fn = partial(_create_object_segment, K=K, threshold=threshold, crf_params=CRF_PARAMS, prefix=prefix, output_dir=output_dir)
    inputs = list(enumerate(sorted(Path(features_root).iterdir())))  # inputs are (index, files) tuples
    _parallel_process(inputs, fn, multiprocessing)


def _create_object_matte(
    inp: Tuple[int, Tuple[str, str]], r: int, prefix: str, output_dir: str, patch_size: int = 16
):
    from pymatting import estimate_alpha_cf, stack_images

    # Load 
    index, (feature_path, segment_path) = inp
    index, path = inp
    try:
        data_dict = torch.load(feature_path, map_location='cpu')
        data_dict.update(torch.load(segment_path, map_location='cpu'))
    except:
        print(f'Problem with index: {index}')
        return

    # Output file
    id = Path(data_dict['file']).stem
    output_file = str(Path(output_dir) / f'{prefix}-matte-{id}.png')
    if Path(output_file).is_file():
        return  # skip because already generated

    # Sizes
    image = _inverse_transform(data_dict['images_resized'].squeeze(0))
    img_np = np.array(image) / 255
    H, W = img_np.shape[:2]
    H_, W_ = (H // patch_size, W // patch_size)

    # Eigenvector
    eigenvector = data_dict['eigenvectors'][1].reshape(H_,W_)
    eigenvector = resize(eigenvector, output_shape=(H, W))  # upscale
    object_segment = (eigenvector > 0).astype(float)
    if 0.5 < np.mean(object_segment).item() < 1.0:  # reverse segment
        object_segment = (1 - object_segment)
    object_segment = get_largest_cc(object_segment)
    trimap = trimap_from_mask(object_segment, r=r)
    alpha = estimate_alpha_cf(img_np, trimap)
    rgba = stack_images(img_np, alpha)

    # Save dict
    # torch.save(rgba, output_file)
    Image.fromarray((rgba * 255).astype(np.uint8)).save(output_file)


def create_object_mattes(
    prefix: str,
    features_root: str = './features_VOC2012',
    segments_root: str = './eigensegments_VOC2012',
    output_dir: str = './mattes_VOC2012',
    r: int = 15,
    multiprocessing: int = 0
):
    """
    Example:
    python extract-segments.py create_object_mattes \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --segments_root ./eigensegments_VOC2012 \
        --output_dir ./mattes_VOC2012 \
    """
    fn = partial(_create_object_matte, r=r, prefix=prefix, output_dir=output_dir)
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
    _parallel_process(inputs, fn, multiprocessing)


def _create_object_mask(
    inp: Tuple[int, Tuple[str, str]], r: int, prefix: str, output_dir: str, patch_size: int = 16
):
    index, (feature_path, segment_path) = inp
    try:

        # Load 
        data_dict = torch.load(feature_path, map_location='cpu')
        data_dict.update(torch.load(segment_path, map_location='cpu'))

        # Output file
        id = Path(data_dict['file']).stem
        output_file = str(Path(output_dir) / f'{prefix}-mask-{id}.png')
        # if Path(output_file).is_file():
        #     return  # skip because already generated

        # Eigenvector
        eigensegment = data_dict['eigensegments_object'][1].numpy()  # get the 2nd smallest eigenvector
        resized_eigensegment = resize(eigensegment, output_shape=data_dict['shape'][-2:])
        resized_eigensegment[:eigensegment.shape[0], :eigensegment.shape[1]] = eigensegment

        # Save dict
        Image.fromarray((eigensegment * 255).astype(np.uint8)).save(output_file)

    except:
        print(f'Problem with index: {index}')
        return


def create_object_masks(
    prefix: str,
    features_root: str = './features_VOC2012',
    segments_root: str = './eigensegments_VOC2012',
    output_dir: str = './mattes_VOC2012',
    r: int = 15,
    multiprocessing: int = 0
):
    """
    Example:
    python extract-segments.py create_object_masks \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --segments_root ./eigensegments_VOC2012 \
        --output_dir ./masks_VOC2012 \
    """
    fn = partial(_create_object_mask, r=r, prefix=prefix, output_dir=output_dir)
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
    _parallel_process(inputs, fn, multiprocessing)


def _create_multilabel_mask(
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


def create_multilabel_masks(
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
    python extract-segments.py create_multilabel_masks \
        --prefix VOC2012-dino_vits16 \
        --features_root ./features_VOC2012 \
        --segments_root ./eigensegments_VOC2012 \
        --output_dir ./multilabel_masks_VOC2012 \
    """
    fn = partial(_create_multilabel_mask, adaptive=adaptive, adaptive_eigenvalue_threshold=adaptive_eigenvalue_threshold, 
                 non_adaptive_num_segments=non_adaptive_num_segments, prefix=prefix, output_dir=output_dir)
    inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
    _parallel_process(inputs, fn, multiprocessing)


def create_semantic_segmentations(
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
    python extract-segments.py create_semantic_segmentations \
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
    pbar = zip(tqdm(feature_files, desc='Loading and pooling features'), multilabel_mask_files)
    for features_file, segmap_file in pbar:
        data_dict = torch.load(str(features_file))
        segmap = np.array(Image.open(segmap_file))
        image_id = data_dict['id']

        # Output file
        output_file = str(Path(output_dir) / f'{prefix}-mask-{image_id}.png')
        if Path(output_file).is_file():
            continue  # skip because already generated

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

        # Loop over features
        image_segment_indices = [index for index in np.unique(segmap) if index != bg_index]
        image_pooled_features = [torch.mean(features * (segmap == index), dim=(0, 1))  # (D, )
                                 for index in image_segment_indices]
       
        # Append
        segmaps.append(segmap)
        image_ids.append(image_id)
        bg_indices.append(bg_index)
        output_files.append(output_file)
        all_pooled_features.append(image_pooled_features)
        all_segment_indices.append(image_segment_indices)

    # Stack
    all_pooled_features_flat = torch.stack([
        feat for image_pooled_features in all_pooled_features for feat in image_pooled_features
    ], dim=0).numpy()
    
    # Kmeans
    n_clusters = num_clusters_excluding_background + (0 if use_background_heuristic else 1)
    kmeans = KMeans(n_clusters=n_clusters)
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


def _create_crf_semantic_segmentations(
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
    if Path(output_file).is_file():
        return  # skip because already generated

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


def create_crf_semantic_segmentations(
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
    python extract-segments.py create_crf_semantic_segmentations \
        --prefix VOC2012-dino_vits16 \
        --semantic_segmentations_root ./semantic_segmentations_VOC2012 \
        --output_dir ./crf_semantic_segmentations_VOC2012 \
    """
    crf_params = (w1, alpha, beta, w2, gamma, it)
    fn = partial(_create_crf_semantic_segmentations, num_clusters=num_clusters, crf_params=crf_params, prefix=prefix, output_dir=output_dir)
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
    for i, (ff, fs) in inputs:
        if i > 20: break

        # Combine
        features_dict = torch.load(ff, map_location='cpu')
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


# def vis_segmentations(
#     images_list: str = '/data_q1_d/machine-learning-datasets/image-captioning/COCO/2014/train2014',
#     images_root: str = '/data_q1_d/machine-learning-datasets/image-captioning/COCO/2014/train2014',
#     segmentations_root: str = './semantic_segmentations_VOC2012',
# ):
#     """
#     Example:
#     streamlit run extract-segments.py vis_segments -- \
#         --images_root ./features_VOC2012 \
#         --segmentations_root ./eigensegments_VOC2012 \
#     """
#     # Streamlit setup
#     import streamlit as st
#     st.set_page_config(layout='wide')

#     # Load
#     inputs = _get_feature_and_segment_inputs(features_root=features_root, segments_root=segments_root)
#     print(f'{len(inputs)=}')

#     # Combine
#     for i, (ff, fs) in inputs:
#         if i > 20: break

#         # Combine
#         features_dict = torch.load(ff, map_location='cpu')
#         segments_dict = torch.load(fs, map_location='cpu')
#         data_dict = defaultdict(list)
#         for k, v in features_dict.items():
#             data_dict[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
#         for k, v in segments_dict.items():
#             data_dict[k] = v[0] if (isinstance(v, list) and len(v) == 1) else v
#         data_dict = dict(data_dict)

#         # Print stuff
#         if i == 0:
#             for k, v in data_dict.items():
#                 st.write(k, type(v), v.shape if torch.is_tensor(v) else (v if isinstance(v, str) else None))

#         # Display stuff
#         img = data_dict['images_resized']
#         image = _inverse_transform(img.squeeze(0))
#         eig_seg = data_dict['eigensegments'].numpy() * 255
#         obj_seg = data_dict['eigensegments_object'].numpy() * 255
#         cols = st.columns(1 + 3 + 3)
#         cols[0].image(image, caption=f'{data_dict["files"][0]} ({i})')
#         cols[1].image(obj_seg[0], caption='obj seg 0')
#         cols[2].image(obj_seg[1], caption='obj seg 1')
#         cols[3].image(obj_seg[2], caption='obj seg 2')
#         cols[4].image(eig_seg[0], caption='eig seg 0')
#         cols[5].image(eig_seg[1], caption='eig seg 1')
#         cols[6].image(eig_seg[2], caption='eig seg 2')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        create_segments=create_segments, 
        create_object_mattes=create_object_mattes,
        create_object_masks=create_object_masks,
        create_multilabel_masks=create_multilabel_masks,
        create_semantic_segmentations=create_semantic_segmentations,
        create_crf_semantic_segmentations=create_crf_semantic_segmentations,
        vis_segments=vis_segments,
    ))
