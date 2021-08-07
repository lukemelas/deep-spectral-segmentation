import os
import sys
import math
import datetime
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

import utils
from model import SimpleModel
from dataset import SimpleDataset, get_transforms


TrainState = namedtuple('TrainStats', ['epoch', 'step', 'best_val'])


@hydra.main(config_path='config', config_name='default')
def main(cfg: DictConfig):

    # Accelerator
    accelerator = Accelerator(fp16=cfg.fp16, cpu=cfg.cpu)

    # Logging
    utils.setup_distributed_print(accelerator.is_local_main_process)
    if cfg.wandb and accelerator.is_local_main_process:
        wandb.init(project='template', name=cfg.name, job_type=cfg.job_type, config=cfg, save_code=True)

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
    if cfg.ema.enabled:
        from timm.utils import ModelEmaV2
        model_ema = ModelEmaV2(model, decay=cfg.ema.decay, device=cfg.ema.device)
        print('Initialized model EMA')
    else:
        model_ema = None
        print('Not using model EMA')

    # Optimizer and scheduler
    optimizer = utils.get_optimizer(cfg, model, accelerator)
    scheduler = utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    best_val = - math.inf
    start_epoch = start_step = 0
    if cfg.checkpoint.resume:
        start_epoch, start_step, best_val = utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)
    train_state = TrainState(epoch=start_epoch, step=start_step, best_val=best_val)

    # Transforms
    train_transform, val_transform = get_transforms(cfg)

    with Timer(prefix="Loading data..."):
        dataset_train = SimpleDataset(root=cfg.data.train.root, transform=train_transform)
        dataset_val = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        dataset_vis = SimpleDataset(root=cfg.data.val.root, transform=val_transform)
        # # Optionally, shuffle the visualization dataset with:
        # indices = np.random.default_rng(seed=cfg.seed).permutation(len(dataset_val))
        # dataset_vis = Subset(dataset_vis, indices=indices)
        dataloader_train = DataLoader(dataset_train, shuffle=True, drop_last=True, **cfg.data.loader)
        dataloader_val = DataLoader(dataset_val, shuffle=False, drop_last=False, **cfg.data.loader)
        dataloader_vis = DataLoader(dataset_vis, shuffle=False, drop_last=False, **cfg.data.loader)
        total_bs = cfg.data.loader.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # Setup
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val)

    # Shared training, evaluation, and visualization args
    kwargs = dict(
        cfg=cfg,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_vis=dataloader_vis,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        model_ema=model_ema,
        train_state=train_state)

    # Evaluation
    if cfg.job_type == 'eval':
        test_stats = evaluate(**kwargs)
        return test_stats['val_loss']

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataset_train):_}')
    print(f'    Dataset val size: {len(dataset_val):_}')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}')
    print(f'    Num epochs: {cfg.epochs:_}')
    print(f'    Batch size per device = {cfg.data.loader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_bs}')
    print(f'    Gradient Accumulation steps = {cfg.gradient_accumulation_steps}')
    print(f'    Total optimization steps = {cfg.max_train_steps}')

    # Training loop
    while True:

        # Visualize (before training)
        visualize(**kwargs, num_batches=1, identifier=f'epoch_{epoch:04d}')

        # Single epoch of training
        train_state = train_one_epoch(**kwargs, train_state=train_state)

        # Save checkpoint on only 1 process
        if accelerator.is_local_main_process:
            print('Saving checkpoint...')
            checkpoint_dict = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': train_state.epoch,
                'steps': train_state.steps,
                'best_val': train_state.best_val,
                'model_ema': model_ema.state_dict() if model_ema else {},
                'cfg': cfg
            }
            torch.save(checkpoint_dict, 'checkpoint-latest.pth')

        # Evaluate
        if (train_state.epoch < 5) or (train_state.epoch) % 5 != 0:
            test_stats = evaluate(**kwargs)
            if accelerator.is_local_main_process:
                if test_stats['top1'] > train_state.best_val:
                    train_state.best_val = test_stats['top1']
                    torch.save(checkpoint_dict, 'checkpoint-best.pth')
                if cfg.wandb:
                    wandb.log(test_stats)
                    wandb.run.summary["best_top1"] = train_state.best_val


def train_one_epoch(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        scheduler: Callable,
        train_state: TrainState,
        model_ema: Optional[object] = None,
        **_unused_kwargs):

    # Train mode
    model.train()
    log_header = f'Epoch: [{train_state.epoch}]'
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    progress_bar = metric_logger.log_every(dataloader_train, cfg.logging.print_freq, header=log_header)

    # Train
    for i, (inputs, target) in enumerate(progress_bar):
        if i >= cfg.get('limit_train_batches', math.inf):
            break

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
        if train_state.step % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if cfg.scheduler.stepwise:
                scheduler.step()
            train_state.step += 1

            # Model EMA
            if model_ema is not None and (train_state.epoch % cfg.ema.update_every) == 0:
                model_ema.update(model)

        # Logging
        log_dict = dict(lr=optimizer.param_groups[0]["lr"],
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
def evaluate(
        *,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloader_val: Iterable,
        accelerator: Accelerator,
        header: str = 'Eval',
        **_unused_kwargs):

    # Eval mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    progress_bar = metric_logger.log_every(dataloader_val, cfg.logging.print_freq, header)

    # Evaluate
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
        *
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

        # Lists for logging
        img_list = [image]
        name_list = ['image']

        # Save only on the main single GPU
        if accelerator.is_local_main_process:
            wandb_log_dict = {}
            for x, name in zip(img_list, name_list):
                x = x.detach().cpu().requires_grad_(False)
                for i, x_i in enumerate(x):
                    ndarr = x_i.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                    pil_img = Image.fromarray(ndarr)
                    filename = f'vis/{name}/img-{i}-{name}-{identifier}.png'
                    Path(filename).parent.mkdir(exist_ok=True, parents=True)
                    pil_img.save(filename)
                    if i < 2:  # log to weights and biases
                        wandb_log_dict[f'img-{i}-{name}'] = [wandb.Image(pil_img)]
            print(f'Saved visualization images (e.g. {filename})')
            if cfg.wandb:
                wandb.log(wandb_log_dict, commit=False)


if __name__ == '__main__':
    main()
