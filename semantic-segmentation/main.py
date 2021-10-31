import os
import sys
import math
import datetime
from contextlib import nullcontext
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image
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

    # Dataset
    if cfg.use_embeddings:
        # Compute embeddings and save to './embeddings'
        if cfg.embedings_dir is None:
            compute_and_save_dense_embeddings(cfg=cfg, accelerator=accelerator)
        # Create dataset with embeddings
        dataset_val = VOCSegmentationWithPseudolabels(
            **cfg.data.val_kwargs, 
            transform=None,  # no transform to evaluate at original resolution
            segments_dir=cfg.get('segments_dir', './segments'),
            features_dir=cfg.get('features_dir', './features'),
        )
    elif cfg.use_segments:
        # Create dataset with segments
        dataset_val = VOCSegmentationWithPseudolabels(
            **cfg.data.val_kwargs, 
            transform=None,  # no transform to evaluate at original resolution
            segments_dir=cfg.segments_dir,
            features_dir=cfg.features_dir,
        )
    else:
        raise NotImplementedError()

    # Dataloaders
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    dataloader_vis = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    
    # Multiple trials
    for i in range(cfg.kmeans.num_trials):

        # Cluster with K-Means
        clusters = kmeans(cfg, dataloader_val, accelerator, seed=(cfg.seed + i))
        
        # # Evaluate
        # eval_stats = kmeans(**kwargs)
        # print(eval_stats)

        # # Visualize
        # visualize(**kwargs, num_batches=1, identifier=f'e-{train_state.epoch}')


def compute_and_save_dense_embeddings(
    *,
    cfg: DictConfig,
    accelerator: Accelerator, 
    val_transform: Callable
):

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
    for i, (inputs, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf): break

        # Forward
        output = model(inputs)

        # Save embeddings...

        raise NotImplementedError()
        


@torch.no_grad()
def kmeans(
        *,
        cfg: DictConfig,
        dataloader_val: Iterable,
        accelerator: Accelerator,
        seed: int,
        **_unused_kwargs):

    # # perform kmeans
    # all_prototypes = all_prototypes.cpu().numpy()
    # all_sals = all_sals.cpu().numpy()
    # n_clusters = n_clusters - 1
    # print('Kmeans clustering to {} clusters'.format(n_clusters))

    # print(colored('Starting kmeans with scikit', 'green'))
    # pca = PCA(n_components=32, whiten=True)
    # all_prototypes = pca.fit_transform(all_prototypes)
    # kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    # prediction_kmeans = kmeans.fit_predict(all_prototypes)

    # # save predictions
    # for i, fname, pred in zip(range(len(val_loader.sampler)), names, prediction_kmeans):
    #     prediction = all_sals[i].copy()
    #     prediction[prediction == 1] = pred + 1
    #     np.save(os.path.join(p['embedding_dir'], fname + '.npy'), prediction)
    #     if i % 300 == 0:
    #         print('Saving results: {} of {} objects'.format(i, len(val_loader.dataset)))


    # Stack all embeddings
    all_features = []
    for i, (inputs, target, embeddings, features) in tqdm(dataloader_val):
        all_features.append(features)
    all_features = torch.stack(features, dim=0).numpy()

    # Perform PCA
    if cfg.kmeans.pca_dim is not None:
        pca = PCA(n_components=32, whiten=True)
        all_features = pca.fit_transform(all_features)

    # Perform kmeans
    n_clusters = cfg.data.num_classes
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_features)




    # Predict clusters


        

@torch.no_grad()
def visualize(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_vis: Iterable,
        accelerator: Accelerator,
        identifier: str = '',
        num_batches: Optional[int] = None,
        **_unused_kwargs):

    raise NotImplementedError()

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_vis, cfg.logging.print_freq, "Vis")

    # Visualize
    for batch_idx, (inputs, target) in enumerate(progress_bar):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Inverse normalization
        Inv = utils.NormalizeInverse(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std)
        image = Inv(inputs).clamp(0, 1)
        vis_dict = dict(image=image)

        # Save images
        wandb_log_dict = {}
        for name, images in vis_dict.items():
            for i, image in enumerate(images):
                pil_image = utils.tensor_to_pil(image)
                filename = f'vis/{name}/p-{accelerator.process_index}-b-{batch_idx}-img-{i}-{name}-{identifier}.png'
                Path(filename).parent.mkdir(exist_ok=True, parents=True)
                pil_image.save(filename)
                if i < 2:  # log to Weights and Biases
                    wandb_filename = f'vis/{name}/p-{accelerator.process_index}-b-{batch_idx}-img-{i}-{name}'
                    wandb_log_dict[wandb_filename] = [wandb.Image(pil_image)]
        if cfg.wandb and accelerator.is_local_main_process:
            wandb.log(wandb_log_dict, commit=False)
    print(f'Saved visualizations to {Path("vis").absolute()}')


if __name__ == '__main__':
    main()
