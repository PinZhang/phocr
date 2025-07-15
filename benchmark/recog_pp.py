import argparse
import json
from collections import defaultdict
from io import BytesIO
from itertools import islice

import numpy as np
from datasets import load_dataset
from paddleocr import TextRecognition
from PIL import Image
from tqdm import tqdm
from utils.scoring import calculate_cer


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def evaluate_recognition(dataset, args):
    ocr = TextRecognition(model_name=args.model_name, enable_mkldnn=False)

    bz = args.batch_size
    load_image = lambda img: np.array(Image.open(BytesIO(img)).convert("RGB"))

    scene_results = defaultdict(lambda: {'predictions': [], 'labels': []})
    save_results = []
    all_predictions = []
    all_labels = []
    for items in tqdm(
        batched(dataset, bz),
        total=len(dataset) // bz + (1 if len(dataset) % bz else 0),
        desc="Processing batches",
    ):
        images = [load_image(item['image']) for item in items]
        scenes = [item['scene'] for item in items]
        md5s = [item['md5'] for item in items]
        labels = [item['text'] for item in items]

        result = ocr.predict(images)
        predictions = [res['rec_text'] for res in result]
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

    with open(f'benchmark/model_predictions/dataset_{args.benchmark_name}_test_v1_ppocrv5.jsonl', 'w') as f:
        for result in save_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return {
        'overall_cer': overall_cer,
        'scene_cers': scene_cers,
        'total_samples': len(all_predictions),
    }


def print_results(results):
    print("=" * 60)
    print("Recognition results")
    print("=" * 60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Overall CER: {results['overall_cer']:.4f}")
    print("-" * 60)
    print("Scene CER:")
    print("-" * 60)

    sorted_scenes = sorted(results['scene_cers'].items(), key=lambda x: x[0])

    for scene, data in sorted_scenes:
        print(f"{scene:20s} | CER: {data['cer']:.4f} | Count: {data['count']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='PP-OCRv5_server_rec')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--benchmark_name', type=str, default='zh_en_rec_bench')

    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(f"puhuilab/{args.benchmark_name}", split="test")

    print(f"Evaluating (samples: {len(dataset)})")
    results = evaluate_recognition(dataset, args)

    print_results(results)


if __name__ == "__main__":
    main()
