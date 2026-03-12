import argparse
import json
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import build_ref_dataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def _resolve_checkpoint(args):
    checkpoint = getattr(args, 'checkpoint', None)
    if checkpoint:
        if os.path.isabs(checkpoint) or os.path.exists(checkpoint):
            return checkpoint
        return os.path.join(args.output_dir, checkpoint)
    return os.path.join(args.output_dir, 'best_model.pth')


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
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, 'vis')
        os.makedirs(args.vis_dir, exist_ok=True)

    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename='test.log',
                 mode='a')
    logger.info(args)

    test_data = build_ref_dataset(args, mode='test')
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    args.model_dir = _resolve_checkpoint(args)
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.checkpoint again!"
            .format(args.model_dir))

    metrics = inference(test_loader, model, args)
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info("=> metrics saved to '{}'".format(metrics_path))


if __name__ == '__main__':
    main()
