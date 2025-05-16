# Params for nn beamsearch : {'R': 70, 'L': 189, 'alpha': 1.13}. Best is trial 33 with value: 0.002148
# Params for phase 1 {'R': 84, 'L': 145, 'alpha': 1.105}. Best is trial 40 with value: 0.001677
# Params for phase 2: {'R': 77, 'L': 171, 'alpha': 1.125}. Best is trial 46 with value: 0.002281
# Params for tuned total tme {'R': 77, 'L': 166, 'alpha': 1.12}. Best is trial 44 with value: 0.003591


# python make_vamana_graphs.py -R 70 -L 189 -alpha 1.13 -base_path  /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -graph_outfile ../data/openai_vamana_graphs/vamana_nn_beamsearch_70_189_1_13


# python make_vamana_graphs.py -R 84 -L 145 -alpha 1.105 -base_path  /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -graph_outfile ../data/openai_vamana_graphs/vamana_phase_1_84_145_1_105


# python make_vamana_graphs.py -R 77 -L 171 -alpha 1.125 -base_path  /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -graph_outfile ../data/openai_vamana_graphs/vamana_phase_2_77_171_1_125


# python make_vamana_graphs.py -R 77 -L 166 -alpha 1.12 -base_path  /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path /users/namanh/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -graph_outfile ../data/openai_vamana_graphs/vamana_total_time_77_166_1_12

python make_vamana_graphs.py -R 12 -L 30 -alpha 1.1 -base_path /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000 -query_path /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/query.i8bin -gt_path /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/msspacev-gt-1M -graph_outfile ../data/msspacev/graph
