import os
import sys
import math
import datetime
from pathlib import Path
from typing import Iterable, Optional
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from accelerate import Accelerator
from contexttimer import Timer
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb

import utils
from model import SimpleModel
from dataset import SimpleDataset


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)

    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(project='template', name=cfg.name, job_type='train', config=cfg, save_code=True)

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Set random seed
    utils.set_seed(cfg.seed)

    # Model
    model = SimpleModel(**cfg.model)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Exponential moving average of model parameters
    model_ema = None
    if cfg.ema.enabled:
        model_ema = ModelEmaV2(model, decay=cfg.ema.decay, device=cfg.ema.device)
        print('Initialized model EMA')

    # Optimizer and scheduler
    optimizer = create_optimizer_v2(model, **cfg.optimizer.kwargs)
    scheduler, _ = create_scheduler(cfg.scheduler, optimizer)

    # Resume from checkpoint
    start_epoch = 0
    if cfg.checkpoint.resume:
        start_epoch = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # Transforms
    crop_size, resize_size = cfg.data.transform.crop_size, cfg.data.transform.resize_size
    train_transform = A.Compose([
        A.RandomResizedCrop(crop_size, crop_size),
        A.HorizontalFlip(),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2()])
    val_transform = A.Compose([
        A.Resize(resize_size, resize_size),
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std),
        ToTensorV2()])

    # Datasets
    with Timer(prefix="Loading data..."):
        dataset_train = SimpleDataset(root=cfg.data.train.root, transform=train_transform)
        dataset_val = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        print(f'Dataset train / val sizes: {len(dataset_train):_} / {len(dataset_val):_}')

    # Dataloader
    dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, **cfg.data.loader)
    dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
    print(f'Dataloader train size: {len(dataloader_train):_} ({dataloader_train.batch_size})')
    print(f'Dataloader val size: {len(dataloader_val):_} ({dataloader_val.batch_size})')

    # Setup
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val)

    # Shared training, evaluation, and visualization args
    kwargs = dict(
        cfg=cfg,
        model=model,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_ema=model_ema)

    # Evaluation
    if cfg.eval:
        test_stats = evaluate(**kwargs)
        return test_stats['top1']

    # Training
    best_top1 = 0
    print(f"Starting training at {datetime.datetime.now()}")
    for epoch in range(start_epoch, cfg.scheduler.epochs):

        # Visualize (before training)
        visualize(**kwargs, epoch=epoch)

        # Single epoch of training
        train_one_epoch(**kwargs, epoch=epoch)

        # Learning rate scheduler - note that if you use a PyTorch scheduler
        # instead of a timm optimizer, you probably just want to call .step()
        # without the epoch argument
        scheduler.step(epoch)

        # Checkpoint
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'model_ema': model_ema.state_dict() if model_ema else {},
            'cfg': cfg
        }

        # Save on only 1 process
        if accelerator.is_local_main_process:
            print('Saving checkpoint...')
            torch.save(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if True:  # (5 < epoch < 275) and (epoch) % 5 != 0:
            test_stats = evaluate(**kwargs, header=f'Val [{epoch}]')
            if accelerator.is_local_main_process:
                if test_stats['top1'] > best_top1:
                    best_top1 = test_stats['top1']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_top1"] = best_top1


def train_one_epoch(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        epoch: int,
        model_ema: Optional[ModelEmaV2] = None,
        **_unused_kwargs):

    # Train mode
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    progress_bar = metric_logger.log_every(train_loader, cfg.logging.print_freq, header=f'Epoch: [{epoch}]')

    # Train
    for i, (input, target) in enumerate(progress_bar):

        # Forward
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
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Model EMA
        if model_ema is not None:
            model_ema.update(model)

        # Logging
        log_dict = dict(lr=optimizer.param_groups[0]["lr"],
                        train_loss=loss_value, train_top1=acc1[0], train_top5=acc5[0])
        metric_logger.update(**log_dict)
        if cfg.wandb and accelerator.is_local_main_process:
            wandb.log(log_dict)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes(device=accelerator.device)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        val_loader: Iterable,
        accelerator: Accelerator,
        header: str = 'Eval',
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = enumerate(metric_logger.log_every(val_loader, cfg.logging.print_freq, header))

    # Evaluate
    for i, (input, target) in progress_bar:

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
    metric_logger.synchronize_between_processes(device=accelerator.device)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        val_loader: Iterable,
        accelerator: Accelerator,
        epoch=0,
        **_unused_kwargs):

    # Load images for visualization
    model.eval()
    batch_idx = min(15, len(val_loader) - 1)  # Let's not get the first batch
    x, _ = next(batch for i, batch in enumerate(iter(val_loader)) if i == batch_idx)
    x = x[:cfg.vis.num_images]  # truncate batch

    # Inverse normalization
    Inv = utils.NormalizeInverse(mean=cfg.data.transform.img_mean, std=cfg.data.transform.img_std)
    x = Inv(x).clamp(0, 1)

    # Lists for logging
    img_list = [x]
    name_list = ['x']

    # Save only on the main single GPU
    if accelerator.is_local_main_process:
        wandb_log_dict = {}
        for x, name in zip(img_list, name_list):
            x = x.detach().cpu().requires_grad_(False)
            for i, x_i in enumerate(x):
                ndarr = x_i.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                pil_img = Image.fromarray(ndarr)
                filename = f'vis/{name}/img-{i}-{name}-{epoch:04d}.png'
                Path(filename).parent.mkdir(exist_ok=True, parents=True)
                pil_img.save(filename)
                if i < 2:  # log to weights and biases
                    wandb_log_dict[f'img-{i}-{name}'] = [wandb.Image(pil_img)]
        print(f'Saved visualization images (e.g. {filename})')
        if cfg.wandb:
            wandb.log(wandb_log_dict, commit=False)


if __name__ == '__main__':
    main()
