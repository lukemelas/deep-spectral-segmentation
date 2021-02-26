import math
import sys
import os
import datetime
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch import einsum
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import timm
from timm.utils import ModelEmaV2, NativeScaler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
from model import SimpleModel
from dataset import SimpleDataset

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Distributed
    utils.init_distributed_mode(cfg)
    device = torch.device('cuda')
    print()

    # Configuration
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    utils.set_seed(cfg.seed)

    # Logging
    if utils.is_main_process():
        wandb.init(project='template', name=cfg.name, job_type='train', config=cfg, save_code=True)

    # Model
    model = SimpleModel()
    model.to(device)

    # Model EMA
    # NOTE: It is important to create EMA model after cuda(), DP wrapper, and
    # AMP but before SyncBN and DDP wrapper
    model_ema = None
    if cfg.ema.enabled:
        model_ema = ModelEmaV2(model, decay=cfg.ema.decay, device=cfg.ema.device)
        print('Initialized model EMA')

    # DDP
    model_without_ddp = model
    if cfg.distributed.enabled:
        print(cfg.distributed.gpu, cfg.distributed, force=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.distributed.gpu])
        model_without_ddp = model.module
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):12d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):12d}')

    # Optimizer
    optimizer = create_optimizer(cfg.optimizer, model_without_ddp)
    lr_scheduler, _ = create_scheduler(cfg.scheduler, optimizer)
    loss_scaler = NativeScaler()

    # Resume from checkpoint
    start_epoch = 0
    if cfg.model.resume:
        checkpoint = torch.load(cfg.model.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print('Loaded model checkpoint')
        if not cfg.eval and {'optimizer', 'lr_scheduler', 'epoch'}.subsubset(set(checkpoint.keys())):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']
            print('Loaded optimizer/scheduler from checkpoint')
            if cfg.ema.enabled and checkpoint['model_ema'] is not None:
                model_ema.load_state_dict(checkpoint['model_ema'])
                print('Loaded model ema from checkpoint')

    # Data
    crop_size = cfg.data.image.crop_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    val_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset_train = SimpleDataset(
        root=cfg.data.train.root,
        transform=train_transform)
    dataset_val = SimpleDataset(
        root=cfg.data.val.root,
        transform=val_transform)

    # Distributed sampler
    if cfg.distributed.enabled:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Dataloader
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, drop_last=True, **cfg.data.loader)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, drop_last=False, **cfg.data.loader)

    # Evaluation
    if cfg.eval:
        raise NotImplementedError()
        # test_stats = evaluate(data_loader_val, model, device)

    # Training
    best_run = 0
    print(f"Starting training at {datetime.datetime.now()}")
    for epoch in range(start_epoch, cfg.scheduler.epochs):
        if cfg.distributed.enabled:
            data_loader_train.sampler.set_epoch(epoch)

        # Single epoch of training
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=cfg.optimizer.clip_grad,
            model_ema=model_ema)

        # Learning rate
        lr_scheduler.step(epoch)

        # Checkpoint
        checkpoint_dict = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'model_ema': model_ema.state_dict() if model_ema else None,
            'cfg': cfg,
        }
        utils.save_on_master(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if True:  # (5 < epoch < 275) and (epoch) % 5 != 0:
            test_stats = evaluate(model, data_loader_val, device)
            if test_stats['acc'] > best_run:
                best_run = test_stats['acc']
                utils.save_on_master(checkpoint_dict, 'checkpoint-best.pth')
            if utils.is_main_process():
                wandb.log(test_stats)
        if utils.is_main_process():
            wandb.run.summary["best_run"] = best_run


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: Optional[float] = None,
                    model_ema: Optional[ModelEmaV2] = None):

    # Train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    # Train
    for i, (input, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward
        with torch.cuda.amp.autocast():
            loss = model(input, target=target)

        # Exit if loss is NaN
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Loss scaling
        # NOTE: is_second_order is for one optimizer (adahessian) only
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()

        # Model EMA
        if model_ema is not None:
            model_ema.update(model)

        # Logging
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if utils.is_main_process():
            wandb.log({'train_loss': loss_value, 'lr': optimizer.param_groups[0]["lr"]})

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val'
    print_freq = 50
    for i, (input, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward
        with torch.cuda.amp.autocast():
            output = model(input)
            loss = model.loss(output=output, target=target)
        torch.cuda.synchronize()

        # Validation metrics
        acc = (torch.argmax(output, dim=1) == target).float().mean()

        # Logging
        batch_size = len(input)
        metric_logger.update(loss=loss.item(), n=batch_size)
        metric_logger.update(acc=acc.item(), n=batch_size)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged val stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
