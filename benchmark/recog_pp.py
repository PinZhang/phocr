import argparse
import json
from collections import defaultdict
from itertools import islice

import cv2
from datasets import load_dataset
from tqdm import tqdm
from utils.scoring import calculate_cer
from rapidocr import RapidOCR

from phocr.utils import LoadImage, Logger

logger = Logger(logger_name=__name__).get_log()


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def main():
    dataset = load_dataset(f"puhuilab/zh_en_rec_bench", split="test")
    print('dataset length: ', len(dataset))

    engine = RapidOCR()
    load_image = LoadImage()

    bz = 1
    elapsed_time = 0
    scene_results = defaultdict(lambda: {'predictions': [], 'labels': []})
    for items in tqdm(
        batched(dataset, bz),
        total=len(dataset) // bz + (1 if len(dataset) % bz else 0),
        desc="Processing batches",
    ):
        images = [load_image(item['image']) for item in items]
        # print('images: ', images[0].shape)
        texts = engine(images[0], use_det=False, use_cls=False).txts
        scenes = [item['scene'] for item in items]
        md5s = [item['md5'] for item in items]
        labels = [item['text'] for item in items]

        for pred, label, scene, md5 in zip(texts, labels, scenes, md5s):
            scene_results[scene]['predictions'].append(pred)
            scene_results[scene]['labels'].append(label)
        
    scene_cers = {}
    for scene, data in scene_results.items():
        scene_cer, _ = calculate_cer(data['predictions'], data['labels'], True)
        scene_cers[scene] = {'cer': scene_cer, 'count': len(data['predictions'])}

    print('scene_cers: ', scene_cers)


if __name__ == "__main__":
    main()
