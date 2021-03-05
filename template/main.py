import math
import sys
import os
import datetime
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

import timm
from timm.utils import ModelEmaV2, NativeScaler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
from model import SimpleModel
from dataset import SimpleDataset


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Distributed
    if cfg.distributed.enabled:
        utils.init_distributed_mode(cfg)
    device = torch.device('cuda')

    # Logging
    if utils.is_main_process():
        wandb.init(project='template', name=cfg.name, job_type='train', config=cfg, save_code=True)

    # Configuration
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    utils.set_seed(cfg.seed)

    # Model
    model = SimpleModel(**cfg.model)
    model.to(device)

    # Model EMA -- note that it is important to create EMA model after cuda(),
    # DP wrapper, and AMP but before SyncBN and DDP wrapper
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

    # Optimizer and scheduler
    optimizer = create_optimizer(cfg.optimizer, model_without_ddp)
    scheduler, _ = create_scheduler(cfg.scheduler, optimizer)

    # Native mixed precision
    if cfg.amp:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    # Resume from checkpoint
    start_epoch = 0
    if cfg.checkpoint.resume:
        checkpoint = torch.load(cfg.checkpoint.resume, map_location='cpu')
        state_dict = checkpoint['model'] if hasattr(checkpoint, 'model') else checkpoint
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        print('Loaded model checkpoint')
        if len(missing_keys):
            print(f' - Missing_keys: {missing_keys}')
        if len(unexpected_keys):
            print(f' - Unexpected_keys: {unexpected_keys}')
        if not cfg.eval and {'optimizer', 'scheduler', 'epoch'}.issubset(set(checkpoint.keys())):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            print('Loaded optimizer/scheduler from checkpoint')
            if cfg.ema.enabled and checkpoint['model_ema'] is not None:
                model_ema.load_state_dict(checkpoint['model_ema'])
                print('Loaded model ema from checkpoint')

    # Transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(cfg.data.transform.crop_size, cfg.data.transform.crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(cfg.data.transform.resize_size, cfg.data.transform.resize_size),
        A.CenterCrop(cfg.data.transform.crop_size, cfg.data.transform.crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2(),
    ])

    # Datasets
    dataset_train = SimpleDataset(root=cfg.data.train.root, transform=train_transform)
    dataset_val = SimpleDataset(root=cfg.data.val.root, transform=val_transform)

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
    print(f'Train dataset / dataloader size: {len(dataset_train)} / {len(data_loader_train)}')
    print(f'Val dataset / dataloader size: {len(dataset_val)} / {len(data_loader_val)}')
    
    # Evaluation
    if cfg.eval:
        test_stats = evaluate(cfg=cfg, model=model, loader=data_loader_val, device=device)
        return test_stats['top1']

    # Training
    best_run = 0
    print(f"Starting training at {datetime.datetime.now()}")
    for epoch in range(start_epoch, cfg.scheduler.epochs):
        if cfg.distributed.enabled:
            data_loader_train.sampler.set_epoch(epoch)

        # Single epoch of training
        train_stats = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=data_loader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_scaler=loss_scaler,
            device=device,
            epoch=epoch,
            model_ema=model_ema)

        # Learning rate scheduler - note that if you use a PyTorch scheduler
        # instead of a timm optimizer, you probably just want to call .step()
        # without the epoch argument
        scheduler.step(epoch)

        # Checkpoint
        checkpoint_dict = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'model_ema': model_ema.state_dict() if model_ema else None,
            'cfg': cfg,
        }
        utils.save_on_master(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if True:  # (5 < epoch < 275) and (epoch) % 5 != 0:
            test_stats = evaluate(cfg=cfg, model=model, loader=data_loader_val, device=device)
            if test_stats['top1'] > best_run:
                best_run = test_stats['top1']
                utils.save_on_master(checkpoint_dict, 'checkpoint-best.pth')
        if utils.is_main_process():
            wandb.run.summary["best_run"] = best_run


def train_one_epoch(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
    epoch: int,
    model_ema: Optional[ModelEmaV2] = None,
    loss_scaler: Optional[object] = None
):

    # Train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Train
    for i, (input, target) in enumerate(metric_logger.log_every(loader, cfg.logging.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward
        if cfg.amp:
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = F.cross_entropy(output, target)
        else:
            output = model(input)
            loss = F.cross_entropy(output, target)

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Exit if loss is NaN
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Loss scaling and backward
        optimizer.zero_grad()
        if cfg.amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(
                loss, optimizer, clip_grad=cfg.optimizer.clip_grad, parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if cfg.optimizer.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.clip_grad)
            optimizer.step()
        torch.cuda.synchronize()

        # Model EMA
        if model_ema is not None:
            model_ema.update(model)

        # Logging
        log_dict = dict(train_loss=loss_value, lr=optimizer.param_groups[0]["lr"], top1=acc1[0], top5=acc5[0])
        metric_logger.update(**log_dict)
        if utils.is_main_process():
            wandb.log(log_dict)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val'
    for i, (input, target) in enumerate(metric_logger.log_every(loader, cfg.logging.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward
        output = model(input)
        loss = F.cross_entropy(output, target)
        torch.cuda.synchronize()

        # Measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        # Logging
        log_dict = dict(val_loss=loss.item(), top1=acc1[0], top5=acc5[0])
        metric_logger.update(**log_dict, n=len(input))  # update with batch size

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
