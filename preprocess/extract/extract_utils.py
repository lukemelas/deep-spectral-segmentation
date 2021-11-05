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
from typing import Callable, Iterable, List, Optional, Tuple, Union, Any
from typing import Optional
import cv2
import denseCRF
import fire
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans
try:
    from sklearnex.cluster import KMeans, DBSCAN
    print('Using sklearnex (accelerated sklearn)')
except:
    from sklearn.cluster import KMeans, DBSCAN


# # Params
# ParamsCRF = namedtuple('ParamsCRF', 'w1 alpha beta w2 gamma it')
# CRF_PARAMS = ParamsCRF(
#     w1    = 6,     # weight of bilateral term  # 10.0,
#     alpha = 40,    # spatial std  # 80,  
#     beta  = 13,    # rgb  std  # 13,  
#     w2    = 3,     # weight of spatial term  # 3.0, 
#     gamma = 3,     # spatial std  # 3,   
#     it    = 5.0,   # iteration  # 5.0, 
# )


def get_model(name):
    if 'dino' in name:
        model = torch.hub.load('facebookresearch/dino:main', name)
        model.fc = torch.nn.Identity()
        val_transform = get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise NotImplementedError()
    model = model.eval()
    return model, val_transform, patch_size, num_heads


def get_transform(name):
    if 'dino' in name:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        raise NotImplementedError()
    return transform


def get_inverse_transform(name):
    if 'dino' in name:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])])
    else:
        raise NotImplementedError()
    return transform
    

class ImagesDataset(Dataset):
    def __init__(self, filenames: str, images_root: Optional[str] = None, transform: Optional[Callable] = None,
                 prepare_filenames: bool = True) -> None:
        self.root = None if images_root is None else Path(images_root)
        self.filenames = sorted(list(set(filenames))) if prepare_filenames else filenames
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.filenames[index]
        full_path = path if self.root is None else str(self.root / path)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, path, index

    def __len__(self) -> int:
        return len(self.filenames)


def get_image_sizes(data_dict: dict):
    P = data_dict['patch_size']
    B, C, H, W = data_dict['shape']
    assert B == 1, 'assumption violated :('
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P
    return (B, C, H, W, P, H_patch, W_patch, H_pad, W_pad)


def load_single_image(path, transform):
    image = Image.open(path)


def get_paired_input_files(dir1, dir2):
    files1 = sorted(Path(dir1).iterdir())
    files2 = sorted(Path(dir2).iterdir())
    assert len(files1) == len(files2)
    return list(enumerate(zip(files1, files2)))


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


def get_feature_and_segment_inputs(features_root, segments_root, segments_name='eigensegments', ext='.pth') -> List[Tuple[int, Tuple[str, str]]]:
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


def parallel_process(inputs: Iterable, fn: Callable, multiprocessing: int = 0):
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

