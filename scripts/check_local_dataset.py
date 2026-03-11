import argparse
import json
import os


def resolve_path(data_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(data_root, path)


def validate_split(json_file, data_root, caption_index):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f'JSON file not found: {json_file}')

    with open(json_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if not isinstance(samples, list) or not samples:
        raise ValueError(f'{json_file} must contain a non-empty list of samples.')

    sample = samples[0]
    required = ['id', 'image', 'mask', 'caption']
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f'{json_file} first sample is missing keys: {missing}')

    if not isinstance(sample['caption'], list) or not sample['caption']:
        raise ValueError(f'{json_file} first sample has invalid caption list.')

    if caption_index < 0 or caption_index >= len(sample['caption']):
        raise IndexError(
            f'caption_index={caption_index} is out of range for first sample in {json_file} '
            f'with {len(sample["caption"])} captions.')

    image_path = resolve_path(data_root, sample['image'])
    mask_path = resolve_path(data_root, sample['mask'])
    for path in [image_path, mask_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path not found: {path}')

    print(f'{json_file}: {len(samples)} samples')
    print(f'  first id: {sample["id"]}')
    print(f'  image: {image_path}')
    print(f'  mask: {mask_path}')
    print(f'  selected caption[{caption_index}]: {sample["caption"][caption_index]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate local CRIS JSON dataset layout.')
    parser.add_argument('--data-root', default='dataset')
    parser.add_argument('--train-file', default='dataset/train.json')
    parser.add_argument('--test-file', default='dataset/test.json')
    parser.add_argument('--caption-index', type=int, default=2)
    args = parser.parse_args()

    validate_split(args.train_file, args.data_root, args.caption_index)
    validate_split(args.test_file, args.data_root, args.caption_index)
    print('Local JSON dataset validation passed.')
