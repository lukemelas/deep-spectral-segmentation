"""
Misc functions, including distributed helpers, mostly from torchvision
"""
import time
import datetime
import random
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Callable, Optional
from PIL import Image
from accelerate import Accelerator
from omegaconf import DictConfig


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_val: Optional[float] = None


def get_optimizer(cfg: DictConfig, model: torch.nn.Module, accelerator: Accelerator) -> torch.optim.Optimizer:
    # Determine the learning rate
    if cfg.optimizer.scale_learning_rate_with_batch_size:
        lr = accelerator.state.num_processes * cfg.data.loader.batch_size * cfg.optimizer.base_lr
        print('lr = {ws} (num gpus) * {bs} (batch_size) * {blr} (base learning rate) = {lr}'.format(
            ws=accelerator.state.num_processes, bs=cfg.data.loader.batch_size, blr=cfg.lr, lr=lr))
    else:  # scale base learning rate by batch size
        lr = cfg.lr
        print('lr = {lr} (absolute learning rate)'.format(lr=lr))
    # Construct optimizer
    if cfg.optimizer.kind == 'torch':
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = getattr(torch.optim, cfg.optimizer.cls)(parameters, lr=lr, **cfg.optimizer.kwargs)
    elif cfg.optimizer.kind == 'timm':
        from timm.optim import create_optimizer_v2
        optimizer = create_optimizer_v2(model, lr=lr, **cfg.optimizer.kwargs)
    elif cfg.optimizer.kind == 'transformers':
        import transformers
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = getattr(transformers, cfg.optimizer.name)(parameters, lr=lr, **cfg.optimizer.kwargs)
    else:
        raise NotImplementedError(f'invalid optimizer config: {cfg.optimizer}')
    return optimizer


def get_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer) -> Callable:
    if cfg.scheduler.kind == 'torch':
        Sch = getattr(torch.optim.lr_scheduler, cfg.scheduler.cls)
        scheduler = Sch(optimizer=optimizer, **cfg.scheduler.kwargs)
        if cfg.scheduler.warmup:
            from warmup_scheduler import GradualWarmupScheduler
            scheduler = GradualWarmupScheduler(  # wrap scheduler with warmup
                optimizer, multiplier=1, total_epoch=cfg.scheduler.warmup, after_scheduler=scheduler)
    elif cfg.scheduler.kind == 'timm':
        from timm.scheduler import create_scheduler
        scheduler, _ = create_scheduler(optimizer=optimizer, args=cfg.scheduler.kwargs)
    elif cfg.scheduler.kind == 'transformers':
        from transformers import get_scheduler
        scheduler = get_scheduler(optimizer=optimizer, **cfg.scheduler.kwargs)
    else:
        raise NotImplementedError(f'invalid scheduler config: {cfg.scheduler}')
    return scheduler


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # reshape
    target = target.reshape(-1)
    output = output.reshape(target.size(0), -1)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device='cuda'):
        """
        Warning: does not synchronize the deque!
        """
        if not using_distributed():
            return
        print(f"device={device}")
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        n = kwargs.pop('n', 1)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, device='cuda'):
        for meter in self.meters.values():
            meter.synchronize_between_processes(device=device)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def set_requires_grad(module, requires_grad=True):
    for p in module.parameters():
        p.requires_grad = requires_grad


def resume_from_checkpoint(cfg, model, optimizer=None, scheduler=None, model_ema=None):
    
    # Resume model state dict
    checkpoint = torch.load(cfg.checkpoint.resume, map_location='cpu')
    if 'model' in checkpoint:
        state_dict, key = checkpoint['model'], 'model'
    else:
        state_dict, key = checkpoint, 'N/A'
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print('Removed "module." from checkpoint state dict')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded model checkpoint key {key} from {cfg.checkpoint.resume}')
    if len(missing_keys):
        print(f' - Missing_keys: {missing_keys}')
    if len(unexpected_keys):
        print(f' - Unexpected_keys: {unexpected_keys}')
    # Resume model ema
    if cfg.ema.use_ema:
        if checkpoint['model_ema']:
            model_ema.load_state_dict(checkpoint['model_ema'])
            print('Loaded model ema from checkpoint')
        else:
            model_ema.load_state_dict(model.parameters())
            print('No model ema in checkpoint; loaded current parameters into model')
    else:
        if 'model_ema' in checkpoint:
            print('Not using model ema, but model_ema found in checkpoint (you probably want to resume it!)')
        else:
            print('Not using model ema, and no model_ema found in checkpoint.')
            
    # Resume optimization state
    if cfg.checkpoint.resume_training and 'train' in cfg.job_type:
        if 'steps' in checkpoint:
            checkpoint['step'] = checkpoint['steps']
        assert {'optimizer', 'scheduler', 'epoch', 'step', 'best_val'}.issubset(set(checkpoint.keys()))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, step, best_val = checkpoint['epoch'] + 1, checkpoint['step'], checkpoint['best_val']
        train_state = TrainState(epoch=epoch, step=step, best_val=best_val)
        print(f'Loaded optimizer/scheduler at epoch {epoch} from checkpoint')
    elif cfg.checkpoint.resume_optimizer_only:
        assert 'optimizer' in set(checkpoint.keys())
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loaded optimizer from checkpoint, but did not load scheduler/epoch')
    else:
        train_state = TrainState()
        print('Did not resume training (i.e. optimizer/scheduler/epoch)')
    
    return train_state


def setup_distributed_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if using_distributed() else 0


def set_seed(seed):
    rank = get_rank()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if using_distributed():
        print(f'Seeding node {rank} with seed {seed}', force=True)
    else:
        print(f'Seeding node {rank} with seed {seed}')


def tensor_to_pil(image: torch.Tensor):
    assert len(image.shape) and image.shape[0] == 3, f"{image.shape=}"
    image = (image.float() * 0.5 + 0.5).clamp(0, 1).detach().cpu().requires_grad_(False)
    ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def albumentations_to_torch(transform):
    def _transform(img, target):
        augmented = transform(image=img, mask=target)
        return augmented['image'], augmented['mask']
    return _transform
