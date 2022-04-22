import os
import pdb
import sys
import math
import datetime
import torch
import hydra
import wandb
import cv2
import numpy as np
from contextlib import nullcontext
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterable, Optional
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from accelerate import Accelerator
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import metrics
import util as utils
from dataset import SegmentationDataset, central_crop


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

    # Datasets
    # NOTE: The batch size must be 1 for test because the masks are different sizes,
    # and evaluation should be done using the mask at the original resolution.
    test_dataloaders = []
    for data_cfg in cfg.data:
        test_dataset = SegmentationDataset(**data_cfg)
        test_dataloader = DataLoader(test_dataset, **{**cfg.dataloader, 'batch_size': 1})
        test_dataloaders.append(test_dataloader)

    # Evaluation
    if cfg.job_type == 'eval':
        for dataloader in test_dataloaders:
            evaluate_predictions(cfg=cfg, accelerator=accelerator, dataloader_val=dataloader)
    else:
        raise NotImplementedError()


@torch.no_grad()
def evaluate_predictions(
        *,
        cfg: DictConfig,
        dataloader_val: Iterable,
        accelerator: Accelerator,
        **_unused_kwargs):

    # Evaluate
    name = dataloader_val.dataset.name
    all_results = []
    pbar = tqdm(dataloader_val, desc=f'Evaluating {name}')
    for i, (images, targets, metadatas) in enumerate(pbar):
        
        # Convert
        targets = targets.squeeze(0)

        # Load predictions 
        id = Path(metadatas['image_file'][0]).stem
        predictions_file = os.path.join(cfg.predictions[name], f'{id}.png')
        preds = np.array(Image.open(predictions_file).convert('L'))  # (H_patch, W_patch)
        assert set(np.unique(preds).tolist()) in [{0, 255}, {0, 1}, {0}], set(np.unique(preds).tolist())
        preds[preds == 255] = 1

        # Resize if segmentation is patchwise
        if cfg.predictions.downsample is not None:
            H, W = targets.shape
            H_pred, W_pred = preds.shape
            H_pad, W_pad = H_pred * cfg.predictions.downsample, W_pred * cfg.predictions.downsample
            H_max, W_max = max(H_pad, H), max(W_pad, W)
            preds = cv2.resize(preds, dsize=(W_max, H_max), interpolation=cv2.INTER_NEAREST)
            preds[:H_pad, :W_pad] = cv2.resize(preds, dsize=(W_pad, H_pad), interpolation=cv2.INTER_NEAREST)

        # Convert, optional center crop, and unsqueeze
        preds = torch.from_numpy(preds)
        if dataloader_val.dataset.crop:
            preds = TF.center_crop(preds, output_size=min(preds.shape))
        preds = torch.unsqueeze(preds, dim=0)
        targets = torch.unsqueeze(targets, dim=0)

        # Compute metrics
        results = metrics.compute_metrics(preds=preds, targets=targets, metrics=['acc', 'iou'])
        all_results.append(results)

    # Aggregate results
    all_results = metrics.list_of_dict_of_lists_to_dict_of_lists(all_results)
    results = metrics.aggregate_metrics(all_results)
    for metric_name, value in results.items():
        print(f'[{name}] {metric_name}: {value}')


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