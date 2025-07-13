import argparse
import json
from collections import defaultdict
from itertools import islice

import cv2
from datasets import load_dataset
from tqdm import tqdm
from utils.scoring import calculate_cer

from phocr import PHOCR, LangRec
from phocr.rec.typings import TextRecInput
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


def evaluate_recognition(dataset, args):
    if args.max_row:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_row, len(dataset))))

    ocr = PHOCR(
        params={
            'Rec.lang_type': LangRec(args.language),
            'Rec.use_beam_search': args.use_beam_search,
            'Rec.device': args.device,
        }
    )

    bz = args.batch_size
    ocr.text_rec.rec_batch_num = bz
    load_image = LoadImage()

    scene_results = defaultdict(lambda: {'predictions': [], 'labels': []})
    save_results = []
    all_predictions = []
    all_labels = []
    elapsed_time = 0
    for items in tqdm(
        batched(dataset, bz),
        total=len(dataset) // bz + (1 if len(dataset) % bz else 0),
        desc="Processing batches",
    ):
        images = [load_image(item['image']) for item in items]
        scenes = [item['scene'] for item in items]
        md5s = [item['md5'] for item in items]
        labels = [item['text'] for item in items]

        result = ocr.text_rec(TextRecInput(img=images))
        predictions = result.txts
        elapsed_time += result.elapse
        for pred, label, scene, md5 in zip(predictions, labels, scenes, md5s):
            all_predictions.append(pred)
            all_labels.append(label)
            scene_results[scene]['predictions'].append(pred)
            scene_results[scene]['labels'].append(label)
            save_results.append({'md5': md5, 'pred': pred})

    overall_cer, _ = calculate_cer(all_predictions, all_labels, True)

    scene_cers = {}
    for scene, data in scene_results.items():
        scene_cer, _ = calculate_cer(data['predictions'], data['labels'], True)
        scene_cers[scene] = {'cer': scene_cer, 'count': len(data['predictions'])}

    # 检测操作系统
    lang = args.language
    if lang == 'ch':
        lang = 'zh_en'
    with open(f'benchmark/model_predictions/dataset_{lang}_test_v1_PHOCR_x86.jsonl', 'w') as f:
        for result in save_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return {
        'overall_cer': overall_cer,
        'scene_cers': scene_cers,
        'total_samples': len(all_predictions),
        'elapsed_time': elapsed_time,
    }


def print_results(results):
    logger.info("=" * 60)
    logger.info("Recognition results")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Overall CER: {results['overall_cer']:.4f}")
    logger.info(f"Elapsed time: {results['elapsed_time']:.4f} seconds")
    logger.info("-" * 60)
    logger.info("Scene CER:")
    logger.info("-" * 60)

    sorted_scenes = sorted(results['scene_cers'].items(), key=lambda x: x[0])

    for scene, data in sorted_scenes:
        logger.info(f"{scene:20s} | CER: {data['cer']:.4f} | Count: {data['count']}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_row', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--use_beam_search', type=bool, default=True)
    parser.add_argument('--benchmark_name', type=str, default='ru_rec_bench')
    parser.add_argument('--language', type=str, default='ru')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    logger.info("Loading dataset...")
    dataset = load_dataset(f"puhuilab/{args.benchmark_name}", split="test")

    logger.info(f"Evaluating (samples: {args.max_row or len(dataset)})")
    results = evaluate_recognition(dataset, args)

    print_results(results)


if __name__ == "__main__":
    main()
