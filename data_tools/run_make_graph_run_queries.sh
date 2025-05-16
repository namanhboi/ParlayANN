# ./make_graph_run_queries -data_type float -dist_func Euclidian -base_path /home/nd433/big-ann-benchmarks/data/open/spacev1b_base.i8bin.crop_nb_1000000 -query_path /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/private_query_30k.bin -gt_path  /home/nd433/big-ann-benchmarks/data/MSSPACEV1B/gt100_private_query_30k.bin -metric recall -k 10 -R 20 -L 100


./make_graph_run_queries -data_type float -dist_func Euclidian -base_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -gt_rank_path  ~/big-ann-benchmarks/data/OpenAIArXiv/openai_gt_rank -metric all -graph_path ~/ParlayANN/data/openai_vamana_graphs/vamana_nn_beamsearch_70_189_1_13 -res_path querying_result_nn_beamsearch_70_189_1_13.csv


./make_graph_run_queries -data_type float -dist_func Euclidian -base_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -gt_rank_path  ~/big-ann-benchmarks/data/OpenAIArXiv/openai_gt_rank -metric all -graph_path ~/ParlayANN/data/openai_vamana_graphs/vamana_phase_1_84_145_1_105 -res_path querying_result_phase_1_84_145_1_105.csv


./make_graph_run_queries -data_type float -dist_func Euclidian -base_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -gt_rank_path  ~/big-ann-benchmarks/data/OpenAIArXiv/openai_gt_rank -metric all -graph_path ~/ParlayANN/data/openai_vamana_graphs/vamana_phase_2_77_171_1_125 -res_path querying_result_phase_2_77_171_1_125.csv


./make_graph_run_queries -data_type float -dist_func Euclidian -base_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_base.bin.crop_nb_100000 -query_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai_query.bin -gt_path ~/big-ann-benchmarks/data/OpenAIArXiv/openai-100K -gt_rank_path  ~/big-ann-benchmarks/data/OpenAIArXiv/openai_gt_rank -metric all -graph_path ~/ParlayANN/data/openai_vamana_graphs/vamana_total_time_77_166_1_12 -res_path querying_result_total_time_77_166_1_12.csv
