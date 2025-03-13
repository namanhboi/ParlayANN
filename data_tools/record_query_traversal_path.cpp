/*
  this file runs a bunch of queries on a prebuilt graph and then write the visited nodes for each query onto an output file

*/

#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <fstream>
#include <filesystem>
#include <iomanip>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/euclidian_point.h"
#include "utils/beamSearch.h"

using namespace parlayANN;

// ./vbase_graph -base_path ../data/sift/sift_learn.fbin -graph_path ../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian


struct QueryTraversalInfo {
  long query_index;
  // first element is index of visited node, second is rank, third is timestamp
  parlay::sequence<std::tuple<unsigned int, size_t, double>> visited_nodes;
};


std::string visited_nodes_to_string(parlay::sequence<std::tuple<unsigned int, size_t, double>> visited_nodes) {
  std::stringstream res;
  res << "\"[";
  for (int i = 0; i < visited_nodes.size(); i++) {
    const auto [index, rank, timestamp] = visited_nodes[i];
    res << "(" << index << "," << rank <<  "," << std::fixed << std::setprecision(15) << timestamp << ")";
    if (i != visited_nodes.size() - 1) res << ",";
    else res << "]\"";
  }
  return res.str();
}


void write_query_traversal_info_to_file(
					parlay::sequence<QueryTraversalInfo> info,
					std::string output_filename = "data.csv") {
  
  std::ofstream output_file;
  output_file.open(output_filename, std::ios_base::app);
  output_file << "query_index,traversal_path" << std::endl;
  for (int i = 0; i < info.size(); i++) {
    QueryTraversalInfo query_info = info[i];
    output_file << query_info.query_index << ",";
    output_file << visited_nodes_to_string(query_info.visited_nodes);
    if (i != info.size() - 1) output_file << std::endl;
  }
  output_file.close();
}



void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}

// saves the index and distance of a visited node on the path to a random query
int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
		"[-base_path <b>] [-graph_path <gF>] [-data_type <tp>] [-dist_func <df>] [-normalize] [-seed] [-k] [-beam_size] [-cut] [-limit] [-degree_limit] [-rerank_factor] [-cut]");
  char* iFile = P.getOptionValue("-base_path"); // path to points
  char* gFile = P.getOptionValue("-graph_path"); // path to already generated graph
  char* qFile = P.getOptionValue("-query_path");
  std::string outputFile = P.getOptionValue("-output_path", "data.csv");
  std::filesystem::path output_path {outputFile};
  if (std::filesystem::exists(output_path)) std::filesystem::remove(output_path);
  
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
  
  std::random_device rd;
  std::mt19937 gen(rd()); // seed
  std::mt19937 gen_seed(seed);
  
  if (vectype == "float") {
    if (dist_func == "Euclidian"){
      if (normalize) abort_with_message("will impl normalization later");



      if (quantize == 8 || quantize == 16) abort_with_message( "Will impl support for quantize later");

      using Point = Euclidian_Point<float>;
      using PR = PointRange<Point>;
      
      PR Points(iFile);
      PR QueryPoints(qFile);
      Graph<unsigned int> G(gFile);
      
      unsigned int start_index = 0;
      const parlay::sequence<unsigned int> starting_points = {start_index};
      parlayANN::QueryParams QP;

      QP.limit = limit != -1 ? limit : (long) G.size();
      QP.rerank_factor = rerank_factor;
      QP.degree_limit = degree_limit != -1 ? degree_limit : (long) G.max_degree();
      QP.k = k;
      QP.cut = cut ;
      QP.beamSize = beam_size;
      parlay::sequence<QueryTraversalInfo> traversal_info = parlay::tabulate(QueryPoints.size(), [&] (long query_index) {
	Point query_point = QueryPoints[query_index];
	parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(Points.size(), [&](long i) {
	  return query_point.distance(Points[i]);
	});

	parlay::sequence<size_t> distances_from_query_to_all_rank = parlay::rank(distances_from_query_to_all);
	
	auto beam_search_res =  beam_search_timestamp(query_point, G, Points, starting_points,  QP);
	auto visited_and_timestamp = beam_search_res.first.second;
	auto visited = visited_and_timestamp.first;
	parlay::sequence<double> visited_timestamp = visited_and_timestamp.second;
	parlay::sequence<size_t> distance_visited_rank = parlay::tabulate(visited.size(), [&] (long i) {
	  return distances_from_query_to_all_rank[visited[i].first];
	});
	parlay::sequence<std::tuple<unsigned int, size_t, double>> vis_info=  parlay::tabulate(visited.size(), [&] (long i) {
	    return std::make_tuple(visited[i].first, distance_visited_rank[i], visited_timestamp[i]);
	});

	QueryTraversalInfo traversal_res = {
	  .query_index = query_index,
	  .visited_nodes = vis_info
	};
	return traversal_res;
	
      });
      write_query_traversal_info_to_file(traversal_info);

      
    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}
