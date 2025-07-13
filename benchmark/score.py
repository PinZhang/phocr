import argparse
import logging

import pandas as pd
from datasets import load_dataset
from utils.scoring import calculate_cer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_name', type=str, default='zh_en_rec_bench')
    parser.add_argument('--pred_file', type=str, required=True)
    args = parser.parse_args()
    logger.info(f'\n========== Scoring {args.benchmark_name} with {args.pred_file} ==========\n')

    dataset = load_dataset(f'puhuilab/{args.benchmark_name}', split='test')
    gt_df = dataset.to_pandas()
    gt_df = gt_df.drop_duplicates(subset=['md5'])
    pred_df = pd.read_json(args.pred_file, lines=True)
    result = []
    for scene, gt_info in gt_df.groupby('scene'):
        gts = []
        preds = []
        for _, row in gt_info.iterrows():
            gts.append(row['text'])
            if row['md5'] not in pred_df['md5'].values:
                preds.append('')
                continue
            preds.append(pred_df[pred_df['md5'] == row['md5']]['pred'].values[0])
        cer, n = calculate_cer(preds, gts, depunctuation=True)
        result.append({'scene': scene, 'gt_num': len(gt_info), 'cer': round(cer, 4), 'char_num': n})

    df = pd.DataFrame(result)
    print(df.to_markdown())
