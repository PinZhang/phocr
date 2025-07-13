echo "JP"
# Baidu jp score
python benchmark/score.py --benchmark_name jp_rec_bench --pred_file ./benchmark/model_predictions/dataset_jp_test_v1_baidu.jsonl
# Ali jp score
python benchmark/score.py --benchmark_name jp_rec_bench --pred_file ./benchmark/model_predictions/dataset_jp_test_v1_ali.jsonl
# PHOCR jp score
python benchmark/score.py --benchmark_name jp_rec_bench --pred_file ./benchmark/model_predictions/dataset_jp_test_v1_PHOCR_cuda.jsonl

# echo "RU"
# Baidu ru score
python benchmark/score.py --benchmark_name ru_rec_bench --pred_file ./benchmark/model_predictions/dataset_ru_test_v1_baidu.jsonl
# Ali ru score
python benchmark/score.py --benchmark_name ru_rec_bench --pred_file ./benchmark/model_predictions/dataset_ru_test_v1_ali.jsonl
# PHOCR ru score
python benchmark/score.py --benchmark_name ru_rec_bench --pred_file ./benchmark/model_predictions/dataset_ru_test_v1_PHOCR_cuda.jsonl

echo "ZH_EN"
# Baidu zh_en score
python benchmark/score.py --benchmark_name zh_en_rec_bench --pred_file ./benchmark/model_predictions/dataset_zh_en_test_v1_baidu.jsonl
# PHOCR zh_en score
python benchmark/score.py --benchmark_name zh_en_rec_bench --pred_file ./benchmark/model_predictions/dataset_zh_en_test_v1_PHOCR_cuda.jsonl

# echo "KO"
# Baidu ko score
python benchmark/score.py --benchmark_name ko_rec_bench --pred_file ./benchmark/model_predictions/dataset_ko_test_v1_baidu.jsonl
# Ali ko score
python benchmark/score.py --benchmark_name ko_rec_bench --pred_file ./benchmark/model_predictions/dataset_ko_test_v1_ali.jsonl
# PHOCR ko score
python benchmark/score.py --benchmark_name ko_rec_bench --pred_file ./benchmark/model_predictions/dataset_ko_test_v1_PHOCR_cuda.jsonl