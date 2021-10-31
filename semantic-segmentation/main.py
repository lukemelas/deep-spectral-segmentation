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
from tqdm import tqdm
from sklearn.cluster import PCA, MiniBatchKMeans
from skimage.color import label2rgb
from matplotlib.cm import get_cmap


import utils
from datasets.voc import VOCSegmentation, VOCSegmentationWithPseudolabels
from model import get_model


@hydra.main(config_path='config', config_name='default')
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
        transform=None,  # no transform to evaluate at original resolution
        segments_dir=cfg.segments_dir,
        features_dir=cfg.features_dir,
    )

    # Dataloaders
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    
    # Multiple trials
    for i in range(cfg.kmeans.num_trials):

        # Cluster with K-Means
        preds = kmeans(cfg=cfg, dataloader_val=dataloader_val, seed=(cfg.seed + i))

        # Evaluate
        eval_stats = kmeans(cfg=cfg, dataloader_val=dataloader_val, preds=preds)
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
        


@torch.no_grad()
def kmeans(
        *,
        cfg: DictConfig,
        dataloader_val: Iterable,
        seed: int,
        **_unused_kwargs):

    # Stack all embeddings
    all_features = []
    for i, (image, target, mask, features, metadata) in tqdm(dataloader_val, desc='Stacking features'):
        all_features.append(features)
    all_features = torch.stack(features, dim=0).numpy()

    # Perform PCA
    if cfg.kmeans.pca_dim is not None:
        pca = PCA(n_components=32, whiten=True)
        all_features = pca.fit_transform(all_features)

    # Perform kmeans
    n_clusters = cfg.data.num_classes
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    preds = kmeans.fit_predict(all_features)    
    return preds
        

@torch.no_grad()
def visualize(
        *,
        cfg: DictConfig,
        preds: Iterable,
        dataloader_val: Iterable):

    # Visualize
    num_vis = 25
    vis_dir = Path('./vis')
    colors = get_cmap('Set3', 20).colors[:,:3]
    pbar = tqdm(zip(preds, dataloader_val), total=num_vis)
    for i, (pred_cluster, (image, target, mask, features, metadata)) in pbar:
        if i >= num_vis:
            break
        image = np.array(image)
        target = np.array(target)
        # Color the mask
        mask[mask == 1] = pred_cluster
        # Overlay mask on image
        image_pred_overlay = label2rgb(label=mask, image=image, colors=colors)
        image_target_overlay = label2rgb(label=target, image=image, colors=colors)
        # Save
        image_id = metadata["id"]
        path_pred = vis_dir / 'pred' / f'{image_id}-pred.png'
        path_target = vis_dir / 'target' / f'{image_id}-target.png'
        path_pred.parent.mkdir(exist_ok=True, parents=True)
        path_target.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(image_pred_overlay).save(str(path_pred))
        Image.fromarray(image_target_overlay).save(str(path_target))
        print(f'Saved visualizations to {Path("vis").absolute()}')


if __name__ == '__main__':
    main()
