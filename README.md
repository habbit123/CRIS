# CRIS: CLIP-Driven Referring Image Segmentation (CVPR2022)

Created by Zhaoqing Wang*, Yu Lu*, Qiang Li*, Xunqiang Tao, Yandong Guo, Mingming Gong and Tongliang Liu

This is an official PyTorch implementation of the [CRIS](https://arxiv.org/pdf/2111.15174)

CLIP-Driven Referring Image Segmentation (CRIS) framework is proposed to transfer the image-level semantic  knowledge of the CLIP model to dense pixel-level referring image segmentation. More specifically, we design a vision-language decoder to propagate fine-grained semantic information from textual representations to each pixel-level activation, which promotes consistency between the two modalities. In addition, we present text-to-pixel contrastive learning to explicitly enforce the text feature similar to the related pixel-level features and dissimilar to the irrelevances.

**:beers:CRIS actives new state-of-the-art performance on RefCOCO, RefCOCO+ and G-Ref with simple framework!**

## Demo
<p align="center">
  <img src="img/demo-CRIS.gif" width="600">
</p>

## Framework
<p align="center">
  <img src="img/pipeline.png" width="600">
</p>

## News
- :wrench: [Jun 6, 2022] Pytorch implementation of CRIS are released.
- :sunny: [Mar 2, 2022] Our paper was accepted by CVPR-2022.



## Main Results

Main results on RefCOCO

| Backbone | val | test A | test B |
| ---- |:-------------:| :-----:|:-----:|
| ResNet50 | 69.52  | 72.72 | 64.70 |
| ResNet101 | 70.47 | 73.18 | 66.10 |

Main results on RefCOCO+

| Backbone | val | test A | test B |
| ---- |:-------------:| :-----:|:-----:|
| ResNet50 | 61.39 |67.10 | 52.48 |
| ResNet101 | 62.27 | 68.08 | 53.68 |

Main results on G-Ref

| Backbone | val | test |
| ---- |:-------------:| :-----:|
| ResNet50 | 59.35 | 59.39 |
| ResNet101 | 59.87 | 60.36 |

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org) (e.g. 1.10.0)
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Quick Start

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

Before training, please login in your wandb via `wandb login` or `wandb login --anonymously`.
To do training of CRIS with 8 GPUs, run:

```
# e.g., Evaluation on the val-set of the RefCOCO dataset
python -u train.py --config config/refcoco/cris_r50.yaml
```

To do evaluation of CRIS with 1 GPU, run:
```
# e.g., Evaluation on the val-set of the RefCOCO dataset
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/cris_r50.yaml \
      --opts TEST.test_split val-test \
             TEST.test_lmdb datasets/lmdb/refcocog_g/val.lmdb
```

## Local dataset adaptation (train/test only)

This repository now supports a local JSON dataset layout with train/test splits only.

Expected local dataset layout:

```none
CRIS/
  ../dataset/
    train.json
    test.json
    train/
      img/
      lbl/
    test/
      img/
      lbl/
```

A sample annotation item looks like:

```json
{
  "id": "apple_black_rot_1",
  "image": "train/img/apple_black_rot_1.jpg",
  "mask": "train/lbl/apple_black_rot_1.png",
  "caption": [
    "the abnormal region",
    "the apple black rot lesion",
    "the circular, sunken lesions with reddish-brown centers and a distinct purple halo on the surface of the apple fruit"
  ]
}
```

Notes:
- The local config is `config/local/cris_r50_local.yaml` and uses `DATA.data_backend: json`.
- `DATA.caption_index` controls which caption is used for both training and testing. The default is `2`, which means `caption[2]`.
- `DATA.mask_binarize: True` converts local grayscale masks to binary masks during training. With your current dataset, all pixels with value `> 0` are treated as foreground.
- Test metrics now align with binary aggregated confusion over all valid pixels and report `IoU`, `Dice`, `Recall`, `mIoU`, and `mACC`.
- `TEST.checkpoint` controls which checkpoint file is evaluated. The default is `best_model.pth` relative to the experiment output directory.
- `train.py` skips validation when `TRAIN.evaluate: False`. In that mode, each epoch still writes `last_model.pth` and mirrors it to `best_model.pth`, so `test.py` can run unchanged.
- The local config disables `wandb` by default and sets `sync_bn: False`, which is the safer default for a remote single-GPU server.
- Image and mask paths in JSON can be relative to `DATA.data_root` or absolute paths.
- Pretrained CLIP weights are still external assets and should be placed under `pretrain/` as referenced by the config.

Linux setup command:

```bash
PYTHON_VERSION=3.10 TORCH_INSTALL_CMD="python -m pip install torch torchvision" bash scripts/setup_linux.sh
```

Lightweight local dataset check:

```bash
python scripts/check_local_dataset.py --data-root ../dataset --train-file ../dataset/train.json --test-file ../dataset/test.json --caption-index 2
```

Linux training command:

```bash
python -u train.py --config config/local/cris_r50_local.yaml
```

Linux test command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u test.py --config config/local/cris_r50_local.yaml
```

If your local dataset paths differ from this layout, override them with `--opts`, for example:

```bash
python -u train.py --config config/local/cris_r50_local.yaml   --opts DATA.data_root /path/to/dataset          DATA.train_file /path/to/dataset/train.json          DATA.test_file /path/to/dataset/test.json          DATA.caption_index 1
```

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{wang2021cris,
  title={CRIS: CLIP-Driven Referring Image Segmentation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```
