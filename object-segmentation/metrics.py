from collections import defaultdict
import torch
import numpy as np


@torch.no_grad()
def compute_metrics(preds, targets, metrics=['f_max', 'acc', 'iou'], threshold=0.5, swap_dims=False, preds_are_soft=False):

    # Move to CPU
    preds = preds.detach()  # .cpu()
    targets = targets.detach()  # .cpu()
    assert len(targets.shape) == 3
    if preds_are_soft:
        assert len(preds.shape) == 4
        soft_preds = torch.softmax(preds, dim=1)[:, (0 if swap_dims else 1)]  # convert to probabilities
        hard_preds = soft_preds > threshold
    else:
        assert 'f_max' not in metrics, 'must have soft preds for f_max'
        assert (len(preds.shape) == 3) 
        assert (preds.dtype == torch.bool) or (preds.dtype == torch.uint8) or (preds.dtype == torch.long)
        assert (preds.max() <= 1) and (preds.min() >= 0)
        soft_preds = [None] * len(preds)
        hard_preds = preds.bool()

    # Compute
    results = defaultdict(list)
    for soft_pred, hard_pred, target in zip(soft_preds, hard_preds, targets):
        if 'f_max' in metrics:
            precision, recall = compute_prs(soft_pred, target, prob_bins=255)
            results['f_max_precision'].append(precision)
            results['f_max_recall'].append(recall)
        if 'f_beta' in metrics:
            precision, recall = precision_recall(target, hard_preds)
            results['f_beta_precision'].append([precision])
            results['f_beta_recall'].append([recall])
        if 'acc' in metrics:
            acc = compute_accuracy(hard_pred, target)
            results['acc'].append(acc)
        if 'iou' in metrics:
            iou = compute_iou(hard_pred, target)
            results['iou'].append(iou)
    return dict(results)


@torch.no_grad()
def aggregate_metrics(totals):
    results = defaultdict(list)
    if 'acc' in totals:
        results['acc'] = mean(totals['acc'])
    if 'iou' in totals:
        results['iou'] = mean(totals['iou'])
    if 'loss' in totals:
        results['loss'] = mean(totals['loss'])
    if 'f_max_precision' in totals and 'f_max_recall' in totals:
        precisions = torch.tensor(totals['f_max_precision'])
        recalls = torch.tensor(totals['f_max_recall'])
        results['f_max'] = F_max(precisions, recalls)
    if 'f_beta_precision' in totals and 'f_beta_recall' in totals:
        precisions = torch.tensor(totals['f_beta_precision'])
        recalls = torch.tensor(totals['f_beta_recall'])
        results['f_beta'] = F_max(precisions, recalls)
    return results


def compute_accuracy(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    return torch.mean((pred == target).to(torch.float)).item()


def compute_iou(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    intersection = torch.sum(pred * (pred == target), dim=[-1, -2]).squeeze()
    union = torch.sum(pred + target, dim=[-1, -2]).squeeze()
    iou = (intersection.to(torch.float) / union).mean()
    iou = iou.item() if (iou == iou) else 0  # deal with nans, i.e. torch.nan_to_num(iou, nan=0.0)
    return iou


def compute_prs(pred, target, prob_bins=255):
    p = []
    r = []
    for split in np.arange(0.0, 1.0, 1.0 / prob_bins):
        if split == 0.0:
            continue
        pr = precision_recall(target, pred > split)
        p.append(pr[0])
        r.append(pr[1])
    return p, r


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)
    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0
    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0
    return precision.item(), recall.item()


def F_scores(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0   # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    f_scores = F_scores(precisions, recalls, betta_sq)
    f_scores = f_scores.mean(dim=0)
    # print('f_scores.shape: ', f_scores.shape)
    # print('torch.argmax(f_scores): ', torch.argmax(f_scores))
    return f_scores.max().item()


def mean(x):
    return sum(x) / len(x)


def list_of_dicts_to_dict_of_lists(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def list_of_dict_of_lists_to_dict_of_lists(LD):
    return {k: [v for dic in LD for v in dic[k]] for k in LD[0]}


def dict_of_lists_to_list_of_dicts(DL):
    return [dict(zip(DL, t)) for t in zip(*DL.values())]
