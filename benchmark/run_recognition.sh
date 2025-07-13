# ch & en
PYTHONPATH=. python benchmark/recognition.py --benchmark_name zh_en_rec_bench --language ch --use_beam_search True --batch_size 8 --device cpu
# ru
PYTHONPATH=. python benchmark/recognition.py --benchmark_name ru_rec_bench --language ru --use_beam_search True --batch_size 8 --device cpu
# ko
PYTHONPATH=. python benchmark/recognition.py --benchmark_name ko_rec_bench --language ko --use_beam_search True --batch_size 8 --device cpu
# jp
PYTHONPATH=. python benchmark/recognition.py --benchmark_name jp_rec_bench --language jp --use_beam_search True --batch_size 8 --device cpu