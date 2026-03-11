import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset import build_ref_dataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def _build_wandb_tags(args):
    tags = getattr(args, 'wandb_tags', None)
    if tags in [None, '']:
        return [args.dataset]
    if isinstance(tags, (list, tuple)):
        return list(tags)
    return [str(tags)]


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    try:
        # wandb
        use_wandb = bool(getattr(args, 'wandb', False))
        if args.rank == 0 and use_wandb:
            wandb.init(job_type="training",
                       mode=getattr(args, 'wandb_mode', 'disabled'),
                       config=dict(args),
                       project=getattr(args, 'wandb_project', 'CRIS'),
                       name=args.exp_name,
                       tags=_build_wandb_tags(args))
        dist.barrier()

        # build model
        model, param_list = build_segmenter(args)
        use_sync_bn = bool(args.sync_bn and args.ngpus_per_node > 1)
        if args.sync_bn and not use_sync_bn and args.rank == 0:
            logger.info(
                'SyncBatchNorm is disabled because only one GPU is available.')
        if use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info(model)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=bool(
                getattr(args, 'find_unused_parameters', True)))

        # build optimizer & lr scheduler
        optimizer = torch.optim.Adam(param_list,
                                     lr=args.base_lr,
                                     weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer,
                                milestones=args.milestones,
                                gamma=args.lr_decay)
        scaler = amp.GradScaler()
        use_val = bool(getattr(args, 'evaluate', True))

        # build dataset
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int(
            (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        train_data = build_ref_dataset(args, mode='train')
        val_loader = None
        if use_val:
            args.batch_size_val = int(args.batch_size_val /
                                      args.ngpus_per_node)
            val_data = build_ref_dataset(args, mode='val')

        # build dataloader
        init_fn = partial(worker_init_fn,
                          num_workers=args.workers,
                          rank=args.rank,
                          seed=args.manual_seed)
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                            shuffle=True)
        train_loader = data.DataLoader(train_data,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       pin_memory=True,
                                       worker_init_fn=init_fn,
                                       sampler=train_sampler,
                                       drop_last=True)
        if use_val:
            val_sampler = data.distributed.DistributedSampler(val_data,
                                                              shuffle=False)
            val_loader = data.DataLoader(val_data,
                                         batch_size=args.batch_size_val,
                                         shuffle=False,
                                         num_workers=args.workers_val,
                                         pin_memory=True,
                                         sampler=val_sampler,
                                         drop_last=False)
        elif args.rank == 0:
            logger.info('Validation is disabled for this run.')

        best_IoU = 0.0 if use_val else None
        # resume
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage: storage.cuda())
                args.start_epoch = checkpoint['epoch']
                if use_val:
                    best_IoU = checkpoint["best_iou"]
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
            else:
                raise ValueError(
                    "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                    .format(args.resume))

        # start training
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            epoch_log = epoch + 1

            # shuffle loader
            train_sampler.set_epoch(epoch_log)

            # train
            train(train_loader, model, optimizer, scheduler, scaler,
                  epoch_log, args)

            # evaluation
            iou = None
            prec_dict = {}
            if use_val:
                iou, prec_dict = validate(val_loader, model, epoch_log, args)

            # save model
            if dist.get_rank() == 0:
                is_best = False
                if use_val and iou >= best_IoU:
                    best_IoU = iou
                    is_best = True
                lastname = os.path.join(args.output_dir, "last_model.pth")
                torch.save(
                    {
                        'epoch': epoch_log,
                        'cur_iou': iou if iou is not None else -1.0,
                        'best_iou': best_IoU if best_IoU is not None else -1.0,
                        'prec': prec_dict,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, lastname)
                bestname = os.path.join(args.output_dir, "best_model.pth")
                if use_val:
                    if is_best:
                        shutil.copyfile(lastname, bestname)
                else:
                    shutil.copyfile(lastname, bestname)

            # update lr
            scheduler.step(epoch_log)
            torch.cuda.empty_cache()

        time.sleep(2)
        if dist.get_rank() == 0 and use_wandb:
            wandb.finish()

        if use_val:
            logger.info("* Best IoU={} * ".format(best_IoU))
        else:
            logger.info('* Validation disabled during training. *')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('* Training time {} *'.format(total_time_str))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
    sys.exit(0)
