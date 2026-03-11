import json
import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class BaseRefDataset(Dataset):
    def __init__(self, dataset, split, mode, input_size, word_length,
                 caption_index=2):
        super(BaseRefDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.caption_index = caption_index
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def _select_caption(self, captions, sample_id):
        if not isinstance(captions, list) or len(captions) == 0:
            raise ValueError('Sample {} has no valid caption list.'.format(
                sample_id))
        if self.caption_index < 0 or self.caption_index >= len(captions):
            raise IndexError(
                'caption_index {} is out of range for sample {} with {} captions.'
                .format(self.caption_index, sample_id, len(captions)))
        return captions[self.caption_index]


class RefDataset(BaseRefDataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length):
        super(RefDataset, self).__init__(dataset=dataset,
                                         split=split,
                                         mode=mode,
                                         input_size=input_size,
                                         word_length=word_length,
                                         caption_index=0)
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.length = self._load_length()
        self.keys = None
        self.env = None

    def _open_db(self):
        if not os.path.exists(self.lmdb_dir):
            raise FileNotFoundError('LMDB path not found: {}'.format(
                self.lmdb_dir))
        return lmdb.open(self.lmdb_dir,
                         subdir=os.path.isdir(self.lmdb_dir),
                         readonly=True,
                         lock=False,
                         readahead=False,
                         meminit=False)

    def _load_length(self):
        env = self._open_db()
        with env.begin(write=False) as txn:
            length = loads_pyarrow(txn.get(b'__len__'))
        env.close()
        return length

    def _init_db(self):
        self.env = self._open_db()
        with self.env.begin(write=False) as txn:
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask
        if self.mode == 'val':
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params

        img = self.convert(img)[0]
        params = {
            'ori_img': ori_img,
            'seg_id': seg_id,
            'mask_dir': mask_dir,
            'inverse': mat_inv,
            'ori_size': np.array(img_size),
            'sents': sents
        }
        return img, params

    def __repr__(self):
        return self.__class__.__name__ + "(" +             f"db_path={self.lmdb_dir}, " +             f"dataset={self.dataset}, " +             f"split={self.split}, " +             f"mode={self.mode}, " +             f"input_size={self.input_size}, " +             f"word_length={self.word_length}"


class JsonRefDataset(BaseRefDataset):
    def __init__(self, data_root, json_file, dataset, split, mode,
                 input_size, word_length, caption_index=2):
        super(JsonRefDataset, self).__init__(dataset=dataset,
                                             split=split,
                                             mode=mode,
                                             input_size=input_size,
                                             word_length=word_length,
                                             caption_index=caption_index)
        self.data_root = data_root
        self.json_file = json_file
        if not os.path.exists(self.json_file):
            raise FileNotFoundError('JSON annotation file not found: {}'.format(
                self.json_file))
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        if not isinstance(self.samples, list):
            raise ValueError('JSON annotation file must contain a list: {}'.format(
                self.json_file))
        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_id = str(sample.get('id', index))
        image_path = self._resolve_path(sample['image'])
        mask_path = self._resolve_path(sample['mask'])
        captions = sample.get('caption', [])
        sent = self._select_caption(captions, sample_id)

        ori_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if ori_img is None:
            raise FileNotFoundError('Image file not found or unreadable: {}'.format(
                image_path))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError('Mask file not found or unreadable: {}'.format(
                mask_path))

        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        if self.mode == 'train':
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask

        if self.mode == 'val':
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_path,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params

        img = self.convert(img)[0]
        params = {
            'ori_img': ori_img,
            'seg_id': sample_id,
            'mask_dir': mask_path,
            'inverse': mat_inv,
            'ori_size': np.array(img_size),
            'sents': [sent]
        }
        return img, params

    def __repr__(self):
        return self.__class__.__name__ + "(" +             f"json_file={self.json_file}, " +             f"data_root={self.data_root}, " +             f"dataset={self.dataset}, " +             f"split={self.split}, " +             f"mode={self.mode}, " +             f"caption_index={self.caption_index}, " +             f"input_size={self.input_size}, " +             f"word_length={self.word_length}"


def build_ref_dataset(args, mode):
    data_backend = getattr(args, 'data_backend', 'lmdb')
    if data_backend == 'json':
        json_attr = '{}_file'.format(mode)
        if not hasattr(args, json_attr) or not getattr(args, json_attr):
            raise ValueError('Missing {} for JSON dataset backend.'.format(
                json_attr))
        split = getattr(args, '{}_split'.format(mode), mode)
        data_root = getattr(args, 'data_root', '')
        caption_index = int(getattr(args, 'caption_index', 2))
        return JsonRefDataset(data_root=data_root,
                              json_file=getattr(args, json_attr),
                              dataset=args.dataset,
                              split=split,
                              mode=mode,
                              input_size=args.input_size,
                              word_length=args.word_len,
                              caption_index=caption_index)
    if data_backend != 'lmdb':
        raise ValueError('Unsupported data backend: {}'.format(data_backend))

    lmdb_attr = '{}_lmdb'.format(mode)
    if not hasattr(args, lmdb_attr) or not getattr(args, lmdb_attr):
        raise ValueError('Missing {} for LMDB dataset backend.'.format(
            lmdb_attr))
    split = getattr(args, '{}_split'.format(mode), mode)
    return RefDataset(lmdb_dir=getattr(args, lmdb_attr),
                      mask_dir=args.mask_root,
                      dataset=args.dataset,
                      split=split,
                      mode=mode,
                      input_size=args.input_size,
                      word_length=args.word_len)
