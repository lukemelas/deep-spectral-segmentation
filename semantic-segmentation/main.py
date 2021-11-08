import os
import sys
import math
import datetime
from contextlib import nullcontext
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from contexttimer import Timer
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import albumentations as A
import albumentations.pytorch
import cv2
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from skimage.color import label2rgb
from matplotlib.cm import get_cmap

import utils
import eval_utils
from datasets.voc import VOCSegmentation, VOCSegmentationWithPseudolabels
from model import get_model


@hydra.main(config_path='config', config_name='eval')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)

    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(name=cfg.name, job_type=cfg.job_type, config=OmegaConf.to_container(cfg), save_code=True, **cfg.wandb_kwargs)
        cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    utils.set_seed(cfg.seed)

    # Create dataset with segments/pseudolabels
    dataset_val = VOCSegmentationWithPseudolabels(
        **cfg.data.val_kwargs, 
        segments_dir=cfg.segments_dir,
        transform=None,  # no transform to evaluate at original resolution
    )

    # If the data is already clustered, then we simply need to evaluate
    if cfg.data_is_already_clustered:
        eval_stats = evaluate(cfg=cfg, dataset_val=dataset_val, preds=None)
        print(eval_stats)
        return
    
    # Multiple trials
    for i in range(cfg.kmeans.num_trials):
        print(f'Starting run {i + 1} of {cfg.kmeans.num_trials}')

        # Cluster with K-Means
        preds = kmeans(cfg=cfg, dataset_val=dataset_val, seed=(cfg.seed + i))
        
        # Visualize
        visualize(cfg=cfg, dataset_val=dataset_val, preds=preds)

        # Evaluate
        eval_stats = evaluate(cfg=cfg, dataset_val=dataset_val, preds=preds)
        print(eval_stats)


def compute_and_save_dense_embeddings(
        *,
        cfg: DictConfig,
        accelerator: Accelerator, 
        val_transform: Callable):

    # Create model
    model = get_model(**cfg.model)
    print('Loaded baseline model')
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Transforms
    crop_size, resize_size = cfg.data.transform.crop_size, cfg.data.transform.resize_size
    val_transform = utils.albumentations_to_torch(transform=A.Compose([
        A.Resize(resize_size, resize_size, interpolation=cv2.INTER_CUBIC), 
        A.CenterCrop(crop_size, crop_size),
        A.pytorch.ToTensor(), 
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std)
    ]))

    # Dataset
    dataset_val = VOCSegmentation(
        **cfg.data.val_kwargs, 
        transform=val_transform, 
        embeddings_dir=cfg.get('embedings_dir', './embeddings')
    )
    
    # Dataloader
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    total_batch_size = cfg.data.loader.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # Setup
    model, dataloader_val = accelerator.prepare(model, dataloader_val)

    # Info
    print(f'***** Starting to extract embeddings at {datetime.datetime.now()} *****')
    print(f'    Dataset val size: {len(dataset_val):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}')
    print(f'    Batch size per device = {cfg.data.loader.batch_size}')
    print(f'    Total batch size = {total_batch_size}')

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_val, cfg.logging.print_freq, header='Computing embeddings')

    # Train
    for i, (image, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf): break

        # Forward
        output = model(image)

        # Save embeddings...

        raise NotImplementedError()
        

def kmeans(
        *,
        cfg: DictConfig,
        dataset_val: Iterable,
        seed: int,
        **_unused_kwargs):

    # Stack all embeddings
    pbar = tqdm(dataset_val, desc='Stacking features')
    all_features = torch.stack([features for (image, target, mask, features, metadata) in pbar], dim=0).numpy()

    # Perform PCA
    if cfg.kmeans.pca_dim:
        pca = PCA(n_components=cfg.kmeans.pca_dim, whiten=True)
        all_features = pca.fit_transform(all_features)

    # Perform kmeans
    n_clusters = cfg.data.num_classes
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    preds = kmeans.fit_predict(all_features)    
    return preds
        

def visualize(
        *,
        cfg: DictConfig,
        dataset_val: Iterable,
        preds: Iterable):

    # Visualize
    num_vis = 40
    vis_dir = Path('./vis')
    colors = get_cmap('tab10', cfg.data.num_classes + 1).colors[:,:3]
    pbar = tqdm(zip(preds, dataset_val), total=num_vis, desc='Saving visualizations: ')
    for i, (pred_cluster, (image, target, mask, features, metadata)) in enumerate(pbar):
        if i >= num_vis:
            break
        image = np.array(image)
        target = np.array(target)
        target[target == 255] = 0  # set the "unknown" regions to background for visualization
        # Color the mask
        mask[mask == 1] = pred_cluster
        # Overlay mask on image
        image_pred_overlay = label2rgb(label=mask, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        image_target_overlay = label2rgb(label=target, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        # Save
        image_id = metadata["id"]
        path_pred = vis_dir / 'pred' / f'{image_id}-pred.png'
        path_target = vis_dir / 'target' / f'{image_id}-target.png'
        path_pred.parent.mkdir(exist_ok=True, parents=True)
        path_target.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray((image_pred_overlay * 255).astype(np.uint8)).save(str(path_pred))
        Image.fromarray((image_target_overlay * 255).astype(np.uint8)).save(str(path_target))
    print(f'Saved visualizations to {vis_dir.absolute()}')


def evaluate(
        *,
        cfg: DictConfig,
        dataset_val: Iterable,
        preds: Optional[Iterable] = None,
        n_clusters: Optional[int] = None):

    # Add background class
    n_classes = cfg.data.num_classes + 1
    if n_clusters is None:
        n_clusters = n_classes

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    # Load all pixel embeddings
    all_preds = np.zeros((len(dataset_val) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(dataset_val) * 500 * 500), dtype=np.float32)
    offset_ = 0

    # Add all pixels to our arrays
    for i in trange(len(dataset_val), desc='Concatenating all predictions'):
        image, target, mask, features, metadata = dataset_val[i]
        # If preds is None, then our data is already clustered. Otherwise, then
        # our mask is binary, and we need to assign all pixels in the mask to
        # the predicted cluster
        if preds is not None:
            pred_cluster = preds[i]
            mask[mask == 1] = pred_cluster
        # Check where ground-truth is valid and append valid pixels to the array
        valid = (target != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_+n_valid] = target[valid]
        # Possibly reshape embedding to match gt.
        if mask.shape != target.shape:
            raise ValueError(f'{mask.shape=} != {target.shape=}')
            # mask = cv2.resize(embedding, target.shape[::-1], interpolation=cv2.INTER_NEAREST)
        # Append the predicted targets in the array
        all_preds[offset_:offset_+n_valid, ] = mask[valid]
        all_gt[offset_:offset_+n_valid, ] = target[valid]
        # Update offset_
        offset_ += n_valid

    # Truncate to the actual number of pixels
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]

    # Do hungarian matching
    num_elems = offset_
    if n_clusters == n_classes:
        print('Using hungarian algorithm for matching')
        match = eval_utils.hungarian_match(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')
    else:
        print('Using majority voting for matching')
        match = eval_utils.majority_vote(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes)
    print(f'Optimal matching: {match}')

    # Remap predictions
    reordered_preds = np.zeros(num_elems, dtype=all_preds.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_preds == int(pred_i)] = int(target_i)

    # TP, FP, and FN evaluation
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

    # Calculate Jaccard index
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Print results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    print('Evaluation of semantic segmentation ')
    print('mIoU is %.2f' % (100*eval_result['mIoU']))
    return eval_result


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()