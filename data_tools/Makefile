include ../algorithms/bench/parallelDefsANN

vec_to_bin : vec_to_bin.cpp
	$(CC) $(CFLAGS) -o vec_to_bin vec_to_bin.cpp $(LFLAGS) 

compute_groundtruth : compute_groundtruth.cpp
	$(CC) $(CFLAGS) -o compute_groundtruth compute_groundtruth.cpp $(LFLAGS) 

compute_range_groundtruth : compute_range_groundtruth.cpp
	$(CC) $(CFLAGS) -o compute_range_groundtruth compute_range_groundtruth.cpp $(LFLAGS) 

crop : crop.cpp
	$(CC) $(CFLAGS) -o crop crop.cpp $(LFLAGS) 

random_sample : random_sample.cpp
	$(CC) $(CFLAGS) -o random_sample random_sample.cpp $(LFLAGS)

record_query_traversal_path : record_query_traversal_path.cpp
	$(CC) $(CFLAGS) -o record_query_traversal_path record_query_traversal_path.cpp $(LFLAGS)

phase_detection : phase_detection.cpp
	$(CC) $(CFLAGS) -o phase_detection phase_detection.cpp $(LFLAGS)

make_graph_run_queries : make_graph_run_queries.cpp
	$(CC) $(CFLAGS) -o make_graph_run_queries make_graph_run_queries.cpp $(LFLAGS)

make_graph_run_queries_gprof : make_graph_run_queries.cpp
	$(CC) -mcx16 -pg -std=c++17 -march=native -DNDEBUG -I . -o make_graph_run_queries_gprof make_graph_run_queries.cpp $(LFLAGS)

merge_graphs: merge_graphs.cpp 
	$(CC) $(CFLAGS) -o merge_graphs merge_graphs.cpp $(LFLAGS)
