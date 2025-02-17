/*
  This file parses a prebuilt graph file and run beam search queries on this
  graph to get data on the number of hops and time spent in
  each phase of beam search traversal. This data is written to a csv file
  

*/


#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/euclidian_point.h"
#include "utils/beamSearch.h"
#include "utils/csvfile.h"

void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}



// this function should be executed sequentially
void record_information_query(
			      long query_index,
			      const parlayANN::PointRange<parlayANN::Euclidian_Point<float>> &Points,
			      const parlayANN::Graph<unsigned int> &G,
			      const parlayANN::QueryParams &QP,
			      const parlay::sequence<unsigned int> starting_points,
			      int top_n, // defines the barrier that separates the 2 phases, nodes that are with the top 100 closest distance to the query node are in phase 2
			      parlay::sequence<double> &time_phase_1, // time spent in phase 1
			      parlay::sequence<double> &time_phase_2, // time spent in phase 2
			      parlay::sequence<int> &num_hops_phase_1, // number of hops in phase 1
			      parlay::sequence<int> &num_hops_phase_2, // number of hops in phase 2, phase 2 includes nodes that are in the top k closest distance to the query node
			      parlay::sequence<int> &node_freq, // frequency of each node to show up in search
			      parlay::sequence<int> &phase_1_freq, // frequency of each node to show up in phase 1
			      parlay::sequence<int> &phase_2_freq) // frequency of each node to show up in phase 2
{

  parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(Points.size(), [&](long i) {
    return Points[query_index].distance(Points[i]);
  });

  parlay::sequence<size_t> distances_from_query_to_all_rank = parlay::rank(distances_from_query_to_all);

  auto r = parlayANN::beam_search_timestamp(Points[query_index], G, Points, starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited = visited_and_timestamp.first;
  parlay::sequence<double> visited_timestamp = visited_and_timestamp.second;

  parlay::sequence<size_t> distance_visted_rank = parlay::tabulate(visited.size(), [&] (long i) {return distances_from_query_to_all_rank[visited[i].first];});
  double phase_1;
  double phase_2;

  int phase_1_hops;
  int phase_2_hops;

  
  for (int i = 0; i < visited.size(); i++) {
    if (distance_visted_rank[i] <= top_n) {
      phase_1 = visited_timestamp[i - 1 >=0 ? i - 1 : 0] - visited_timestamp[0];
      phase_2 = visited_timestamp[visited.size() - 1] - visited_timestamp[i];
      phase_1_hops = i - 1 >= 0 ? i - 1 : 0;
      phase_2_hops = visited.size() - i;
      break;
    }
  }
  time_phase_1[query_index] = phase_1;
  time_phase_2[query_index] = phase_2;
  num_hops_phase_1[query_index] = phase_1_hops;
  num_hops_phase_2[query_index] = phase_2_hops;
  for (int i = 0; i < visited.size(); i++) {
    node_freq[visited[i].first]++;
    if (distance_visted_rank[i] <= top_n) {
      phase_2_freq[visited[i].first]++;
    } else {
      phase_1_freq[visited[i].first]++;
    }
  }
  return;
  

}


void write_to_csv(std::string csv_filename,
		  parlay::sequence<double> &time_phase_1, // time spent in phase 1
		  parlay::sequence<double> &time_phase_2, // time spent in phase 2
		  parlay::sequence<int> &num_hops_phase_1, // number of hops in phase 1
		  parlay::sequence<int> &num_hops_phase_2, // number of hops in phase 2, phase 2 includes nodes that are in the top k closest distance to the query node
		  parlay::sequence<int> &node_freq, // frequency of each node to show up in search
		  parlay::sequence<int> &phase_1_freq, // frequency of each node to show up in phase 1
		  parlay::sequence<int> &phase_2_freq // frequency of each node to show up in phase 
		  ) {
  parlayANN::csvfile csv(csv_filename);

  csv << "point_index"
  << "query_time_phase_1"
  << "query_time_phase_2"
  << "query_num_hops_phase_1"
  << "query_num_hops_phase_2"
  << "node_freq"
  << "phase_1_freq"
  << "phase_2_freq" << parlayANN::endrow;
  
  for (long i = 0; i < time_phase_1.size(); i++) {
    csv << i << time_phase_1[i] << time_phase_2[i] << num_hops_phase_1[i] << num_hops_phase_2[i]
    << node_freq[i] << phase_1_freq[i] << phase_2_freq[i] << parlayANN::endrow;
  }
  csv << parlayANN::endrow;
  csv << parlayANN::endrow;
}


int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
		"[-base_path <b>] [-graph_path <gF>] [-data_type <tp>] [-dist_func <df>] [-normalize] [-seed] [-k] [-beam_size] [-cut] [-limit] [-degree_limit] [-rerank_factor] [-cut] [-top_n] [-num_query_points] [-res_filename]");
  char* iFile = P.getOptionValue("-base_path"); // path to points
  char* gFile = P.getOptionValue("-graph_path"); // path to already generated graph
  std::string vectype = P.getOptionValue("-data_type");
  std::string dist_func = P.getOptionValue("-dist_func");
  bool normalize = P.getOption("-normalize");
  int quantize = P.getOptionIntValue("-quantize_bits", 0);
  int seed = P.getOptionIntValue("-seed", -1);

  long limit = P.getOptionLongValue("-limit", -1);
  int rerank_factor = P.getOptionIntValue("-rerank_factor", 100);
  long degree_limit = P.getOptionLongValue("-dergee_limit", -1);
  int k = P.getOptionIntValue("-k", 10);
  int beam_size = P.getOptionIntValue("-beam_size", 10);
  double cut = P.getOptionDoubleValue("-cut", 1.35);
  int top_n = P.getOptionIntValue("-top_n", 100);
  char* rFile = P.getOptionValue("-res_filename");
  std::string csv_filename = std::string(rFile);
  std::random_device rd;
  std::mt19937 gen(rd()); // seed
  std::mt19937 gen_seed(seed);
  
  if (vectype == "float") {
    if (dist_func == "Euclidian"){
      parlayANN::PointRange<parlayANN::Euclidian_Point<float>> Points(iFile);
      if (normalize) abort_with_message("will impl normalization later");
      
      parlayANN::Graph<unsigned int> G(gFile);

      if (quantize == 8 || quantize == 16) abort_with_message( "Will impl support for quantize later");

      size_t num_query_points = P.getOptionIntValue("-num_query_points", Points.size());
      using Point = parlayANN::Euclidian_Point<float>;
      using PR = parlayANN::PointRange<Point>;
      
      std::uniform_int_distribution<> distr(0, Points.size() - 1);

      parlayANN::QueryParams QP;

      QP.limit = limit != -1 ? limit : (long) G.size();
      QP.rerank_factor = rerank_factor;
      QP.degree_limit = degree_limit != -1 ? degree_limit : (long) G.max_degree();
      QP.k = k;
      QP.cut = cut ; 
      QP.beamSize = beam_size;
	

      unsigned int start_index = 0;
      parlay::sequence<unsigned int> starting_points = {start_index};


      parlay::sequence<double> time_phase_1(Points.size(), 0.0); // time spent in phase 1
      parlay::sequence<double> time_phase_2(Points.size(), 0.0); // time spent in phase 2
      parlay::sequence<int> num_hops_phase_1(Points.size(), 0); // number of hops in phase 1
      parlay::sequence<int> num_hops_phase_2(Points.size(), 0); // number of hops in phase 2, phase 2 includes nodes that are in the top k closest distance to the query node
      parlay::sequence<int> node_freq(Points.size(), 0); // frequency of each node to show up in search
      parlay::sequence<int> phase_1_freq(Points.size(), 0); // frequency of each node to show up in phase 1
      parlay::sequence<int> phase_2_freq(Points.size(), 0); // frequency of each node to show up in phase 2

      std::cout << "hello" << std::endl;
      
      if (num_query_points == Points.size()) {
	parlay::parallel_for(0, num_query_points, [&] (long i){
	  return record_information_query(i, Points, G, QP, starting_points, top_n, time_phase_1, time_phase_2, num_hops_phase_1, num_hops_phase_2, node_freq, phase_1_freq, phase_2_freq);
	});
	
	write_to_csv(csv_filename, time_phase_1, time_phase_2, num_hops_phase_1, num_hops_phase_2, node_freq, phase_1_freq, phase_2_freq);
	
	std::cout << "finished" << std::endl;
	
	
      } else {
	for (int oo = 0; oo < num_query_points; oo++) {
      
	  long random_index;

	  if (seed == -1) random_index = distr(gen);
	  else random_index = distr(gen_seed);

	}
      }

      
    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}

