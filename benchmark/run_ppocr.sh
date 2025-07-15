# ch & en
PYTHONPATH=. python benchmark/recog_pp.py --benchmark_name zh_en_rec_bench --model_name PP-OCRv5_server_rec --batch_size 8
# ru
PYTHONPATH=. python benchmark/recog_pp.py --benchmark_name ru_rec_bench --model_name eslav_PP-OCRv5_mobile_rec --batch_size 8
# ko
PYTHONPATH=. python benchmark/recog_pp.py --benchmark_name ko_rec_bench --model_name korean_PP-OCRv5_mobile_rec --batch_size 8
# jp
PYTHONPATH=. python benchmark/recog_pp.py --benchmark_name jp_rec_bench --model_name PP-OCRv5_server_rec --batch_size 8