# ./make_graph_run_queries -data_type float -dist_func Euclidian -base_path /home/nd433/big-ann-benchmarks/data/open/spacev1b_base.i8bin.crop_nb_1000000 -query_path /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/private_query_30k.bin -gt_path  /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/gt100_private_query_30k.bin -metric recall -k 10 -R 20 -L 100


time ./make_graph_run_queries -data_type float -dist_func Euclidian -base_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -gt_rank_path  ~/big-ann-benchmarks/data/OpenAIArXiv/openai_gt_ranking -metric all -R 32 -L 100 -k 10 -alpha 1.2
