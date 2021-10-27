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
from torchvision.datasets import VOCSegmentation

import utils
from model import SimpleModel


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

    # Model
    model = SimpleModel(**cfg.model)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Transforms
    crop_size, resize_size = cfg.data.transform.crop_size, cfg.data.transform.resize_size
    train_transform = val_transform = utils.albumentations_to_torch(transform=A.Compose([
        A.Resize(resize_size, resize_size), A.CenterCrop(crop_size, crop_size),
        A.ToTensor(), A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std)
    ]))

    # Datasets
    dataset_train = VOCSegmentation(**cfg.data.train_kwargs, transform=train_transform)
    dataset_val = VOCSegmentation(**cfg.data.val_kwargs, transform=val_transform)
    dataset_vis = VOCSegmentation(**cfg.data.val_kwargs, transform=val_transform)
    
    # Dataloaders
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    dataloader_vis = DataLoader(dataset_vis, shuffle=False, drop_last=False, **{**cfg.data.loader, 'batch_size': 16})
    total_batch_size = cfg.data.loader.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # Setup
    model, dataloader_train, dataloader_val = accelerator.prepare(model, dataloader_train, dataloader_val)

    # Shared training, evaluation, and visualization args
    kwargs = dict(
        cfg=cfg,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_vis=dataloader_vis,
        accelerator=accelerator)

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataset_train):_}')
    print(f'    Dataset val size: {len(dataset_val):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}')
    print(f'    Batch size per device = {cfg.data.loader.batch_size}')

    # Multiple trials
    for _ in range(cfg.num_trials):

        # Load or compute dense embeddings
        if cfg.embeddings is not None:
            compute_and_save_dense_embeddings(**kwargs)
        
        # Cluster with K-Means
        clusters = kmeans(**kwargs)
        
        # # Evaluate
        # eval_stats = kmeans(**kwargs)
        # print(eval_stats)

        # # Visualize (before training)
        # visualize(**kwargs, num_batches=1, identifier=f'e-{train_state.epoch}')


def compute_and_save_dense_embeddings(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_train: Iterable,
        accelerator: Accelerator,
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_train, cfg.logging.print_freq, header='Computing embeddings:')

    # Train
    for i, (inputs, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf): break

        # Forward
        output = model(inputs)
        loss = F.cross_entropy(output, target)

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Exit if loss is NaN
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Loss scaling and backward
        accelerator.backward(loss)

        # Gradient accumulation, optimizer step, scheduler step
        if i % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if cfg.scheduler.stepwise:
                scheduler.step()
            train_state.step += 1

            # Model EMA
            if model_ema is not None and (train_state.epoch % cfg.ema.update_every) == 0:
                model_ema.update()

        # Logging
        log_dict = dict(lr=optimizer.param_groups[0]["lr"], step=train_state.step,
                        train_loss=loss_value, train_top1=acc1[0], train_top5=acc5[0])
        metric_logger.update(**log_dict)
        if cfg.wandb and accelerator.is_local_main_process:
            wandb.log(log_dict)

    # Scheduler
    if not cfg.scheduler.stepwise:
        scheduler.step()

    # Epoch complete
    train_state.epoch += 1

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    print("Averaged stats:", metric_logger)
    return train_state


@torch.no_grad()
def kmeans(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_val: Iterable,
        accelerator: Accelerator,
        model_ema: Optional[object] = None,
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_val, cfg.logging.print_freq, header="Val")
    eval_context = model_ema.average_parameters if cfg.ema.use_ema else nullcontext

    # Evaluate
    with eval_context():
        for i, (inputs, target) in enumerate(progress_bar):
            if i >= cfg.get('limit_val_batches', math.inf):
                break

            # Forward
            output = model(inputs)
            loss = F.cross_entropy(output, target)
            torch.cuda.synchronize()

            # Measure accuracy
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # Logging
            log_dict = dict(val_loss=loss.item(), top1=acc1[0], top5=acc5[0])
            metric_logger.update(**log_dict, n=len(inputs))  # update with batch size

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
