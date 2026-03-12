import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger

from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


@dataclass
class Agg:
    tp_fg: int = 0
    fp_fg: int = 0
    fn_fg: int = 0
    tn: int = 0
    valid_pixels: int = 0

    def update(self, other):
        self.tp_fg += other.tp_fg
        self.fp_fg += other.fp_fg
        self.fn_fg += other.fn_fg
        self.tn += other.tn
        self.valid_pixels += other.valid_pixels



def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0



def _to_sent_text(sent):
    if isinstance(sent, (list, tuple)):
        if len(sent) != 1:
            raise RuntimeError('Expected a single sentence entry, got: {}'.format(sent))
        return sent[0]
    return sent



def _load_binary_mask(mask_dir, args):
    mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError('Mask file not found or unreadable: {}'.format(mask_dir))
    if bool(getattr(args, 'mask_binarize', False)):
        threshold = int(getattr(args, 'mask_positive_threshold', 0))
        return (mask > threshold).astype(np.uint8)
    return (mask > 0).astype(np.uint8)



def confusion_binary_fg(pred_bin: np.ndarray, gt_bin: np.ndarray,
                        valid: np.ndarray) -> Agg:
    if pred_bin.shape != gt_bin.shape or pred_bin.shape != valid.shape:
        raise RuntimeError(
            f'Shape mismatch: pred {pred_bin.shape}, gt {gt_bin.shape}, valid {valid.shape}')

    p = pred_bin.astype(bool) & valid
    g = gt_bin.astype(bool) & valid
    v = valid

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_and(~g, v)).sum())
    fn = int(np.logical_and(np.logical_and(~p, v), g).sum())
    tn = int(np.logical_and(np.logical_and(~p, v), np.logical_and(~g, v)).sum())
    return Agg(tp_fg=tp, fp_fg=fp, fn_fg=fn, tn=tn, valid_pixels=int(v.sum()))



def metrics_from_agg(agg: Agg):
    tp_fg, fp_fg, fn_fg, tn = agg.tp_fg, agg.fp_fg, agg.fn_fg, agg.tn

    tp_bg = tn
    fp_bg = fn_fg
    fn_bg = fp_fg

    iou_bg = _safe_div(tp_bg, tp_bg + fp_bg + fn_bg)
    iou_fg = _safe_div(tp_fg, tp_fg + fp_fg + fn_fg)

    acc_bg = _safe_div(tp_bg, tp_bg + fn_bg)
    acc_fg = _safe_div(tp_fg, tp_fg + fn_fg)

    miou = 0.5 * (iou_bg + iou_fg)
    macc = 0.5 * (acc_bg + acc_fg)

    return {
        'IoU_bg': iou_bg,
        'IoU_fg': iou_fg,
        'mIoU': miou,
        'Acc_bg': acc_bg,
        'Acc_fg': acc_fg,
        'mAcc': macc,
        'valid_pixels': float(agg.valid_pixels),
    }



def _reduce_agg(agg: Agg, device):
    if not dist.is_available() or not dist.is_initialized():
        return agg
    tensor = torch.tensor([
        agg.tp_fg, agg.fp_fg, agg.fn_fg, agg.tn, agg.valid_pixels
    ], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return Agg(tp_fg=int(tensor[0].item()),
               fp_fg=int(tensor[1].item()),
               fn_fg=int(tensor[2].item()),
               tn=int(tensor[3].item()),
               valid_pixels=int(tensor[4].item()))



def _format_metrics(metrics):
    return (
        'IoU_bg={:.2f}  IoU_fg={:.2f}  mIoU={:.2f}  '
        'Acc_bg={:.2f}  Acc_fg={:.2f}  mAcc={:.2f}  valid_pixels={:.0f}'
    ).format(100. * metrics['IoU_bg'],
             100. * metrics['IoU_fg'],
             100. * metrics['mIoU'],
             100. * metrics['Acc_bg'],
             100. * metrics['Acc_fg'],
             100. * metrics['mAcc'],
             metrics['valid_pixels'])



def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    use_wandb = bool(getattr(args, 'wandb', False)) and wandb.run is not None

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        with amp.autocast():
            pred, target, loss = model(image, text, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0] and use_wandb:
                wandb.log(
                    {
                        'time/batch': batch_time.val,
                        'time/data': data_time.val,
                        'training/lr': lr.val,
                        'training/loss': loss_meter.val,
                        'training/iou': iou_meter.val,
                        'training/prec@50': pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args):
    agg = Agg()
    model.eval()
    time.sleep(2)
    pred_threshold = float(getattr(args, 'pred_threshold', 0.35))
    device = next(model.parameters()).device
    for imgs, texts, param in val_loader:
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred_bin = np.array(pred > pred_threshold, dtype=np.uint8)
            gt_bin = _load_binary_mask(mask_dir, args)
            valid = np.ones_like(gt_bin, dtype=bool)
            agg.update(confusion_binary_fg(pred_bin, gt_bin, valid))

    agg = _reduce_agg(agg, device)
    metrics = metrics_from_agg(agg)
    logger.info('Evaluation: Epoch=[{}/{}]  {}'.format(
        epoch, args.epochs, _format_metrics(metrics)))
    return metrics['mIoU'], metrics


@torch.no_grad()
def inference(test_loader, model, args):
    agg = Agg()
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    pred_threshold = float(getattr(args, 'pred_threshold', 0.35))
    for img, param in tbar:
        img = img.cuda(non_blocking=True)
        mask_dir = param['mask_dir'][0]
        gt_bin = _load_binary_mask(mask_dir, args)
        if args.visualize:
            seg_id = param['seg_id'][0]
            if hasattr(seg_id, 'cpu'):
                seg_id = seg_id.cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            ori_img = param['ori_img'][0]
            if hasattr(ori_img, 'cpu'):
                ori_img = ori_img.cpu().numpy()
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=ori_img)
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=np.array(gt_bin * 255, dtype=np.uint8))
        for sent in param['sents']:
            sent_text = _to_sent_text(sent)
            text = tokenize(sent_text, args.word_len, True)
            text = text.cuda(non_blocking=True)
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred_bin = np.array(pred > pred_threshold, dtype=np.uint8)
            valid = np.ones_like(gt_bin, dtype=bool)
            agg.update(confusion_binary_fg(pred_bin, gt_bin, valid))
            if args.visualize:
                pred_vis = np.array(pred_bin * 255, dtype=np.uint8)
                sent_slug = '_'.join(sent_text.split(' '))
                pred_name = '{}-{}.png'.format(seg_id, sent_slug)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name),
                            img=pred_vis)
    logger.info('=> Metric Calculation <=')
    metrics = metrics_from_agg(agg)
    logger.info(_format_metrics(metrics))
    return metrics
