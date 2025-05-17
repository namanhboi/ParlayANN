/*
  this file runs a bunch of queries on a prebuilt graph and then write the visited nodes for each query onto an output file

  if the res_path is provided then it will also run a beamsearch sweep to check recall of graph

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
#include "utils/stats.h"
#include "utils/parse_results.h"
#include "utils/check_nn_recall.h"
using namespace parlayANN;

// ./vbase_graph -base_path ../data/sift/sift_learn.fbin -graph_path ../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian


template<typename indexType, typename Point>
struct VisitedNodeInfo {
  indexType node_index;
  parlay::sequence<std::pair<indexType, typename Point::distanceType>> frontier;
  double distance_to_query;
  size_t distance_to_query_rank;
  double timestamp_visit;
};


template<typename indexType, typename Point>
struct QueryTraversalFrontierHistoryInfo {
  long query_index;

  parlay::sequence<VisitedNodeInfo<indexType, Point>> visited_nodes;
  
  // number of distance comparisons
  size_t num_dist_cmps;
};


struct QueryTraversalInfo {
  long query_index;
  // first element is index of visited node, second is rank, third is timestamp
  parlay::sequence<std::tuple<unsigned int, size_t, double>> visited_nodes;
};


//   res << "query_index,node_index,distance_to_query,distance_to_query_rank,timestamp_visit,frontier\n";
template<typename indexType, typename Point>
std::string query_traversal_frontier_history_info_to_string(QueryTraversalFrontierHistoryInfo<indexType, Point> &info) {
  std::stringstream res("");
  for (auto &visited_node_info : info.visited_nodes) {
    res << info.query_index << ","
    << visited_node_info.node_index << ","
    << visited_node_info.distance_to_query << ","
    << visited_node_info.distance_to_query_rank << ","
    << std::fixed << std::setprecision(15) << visited_node_info.timestamp_visit << ",\"[";
    for (int i = 0; i < visited_node_info.frontier.size(); i++) {
      if (i != visited_node_info.frontier.size() - 1) {
	res << visited_node_info.frontier[i].first << ",";
      } else {
	res << visited_node_info.frontier[i].first << "]\"\n";
      }
    }
  }
  return res.str();
}

template<typename indexType, typename Point>
void write_query_traversal_frontier_history_info_to_file(
							 parlay::sequence<QueryTraversalFrontierHistoryInfo<indexType, Point>> traversal_info,
							 std::string output_filename) {
  std::ofstream output_file;
  output_file.open(output_filename, std::ios_base::app);
  output_file << "query_index,node_index,distance_to_query,distance_to_query_rank,timestamp_visit,frontier\n";
  for (int i = 0; i < traversal_info.size(); i++) {
    output_file << query_traversal_frontier_history_info_to_string(traversal_info[i]);
  }
  output_file.close();
}


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
  char* resFile = P.getOptionValue("-res_path");
  char* gtFile = P.getOptionValue("-gt_path");
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
  double cut = P.getOptionDoubleValue("-cut", 1.0);

  bool frontier_history = P.getOption("-frontier_history");


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
      if (resFile != nullptr) {
	groundTruth<unsigned int> GT (gtFile);
	std::string graph_name = P.getOptionValue("-frontier_history", "Graph");
	std::string params = "params";
	auto [avg_deg, max_deg] = graph_stats_(G);
	double idx_time = 0.0;
	Graph_ G_(graph_name, params, G.size(), avg_deg, max_deg, idx_time);
	
	search_and_parse(G_, G, Points, QueryPoints, GT, resFile, k, true);
	
      }


      if (!frontier_history) {
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
      } else {	
	parlay::sequence<QueryTraversalFrontierHistoryInfo<unsigned int, Point>> traversal_info = parlay::tabulate(QueryPoints.size(), [&] (long query_index) {
	  Point query_point = QueryPoints[query_index];
	  parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(Points.size(), [&](long i) {
	    return query_point.distance(Points[i]);
	  });

	  parlay::sequence<size_t> distances_from_query_to_all_rank = parlay::rank(distances_from_query_to_all);

	  auto beam_search_res = beam_search_timestamp_and_candidate_list(query_point, G, Points, starting_points,  QP);
	
	  beamSearchInfo beam_search_info = beam_search_res.first;
	  size_t num_dist_cmps = beam_search_res.second;

	  parlay::sequence<size_t> distance_visited_rank = parlay::tabulate(beam_search_info.not_sorted_visited.size(), [&] (long i) {
	    return distances_from_query_to_all_rank[beam_search_info.not_sorted_visited[i].first];
	  });

	  parlay::sequence<VisitedNodeInfo<unsigned int, Point>> visited_node_info = parlay::tabulate(beam_search_info.not_sorted_visited.size(), [&] (long i) {
	    VisitedNodeInfo<unsigned int, Point> info;
	    info.node_index = beam_search_info.not_sorted_visited[i].first;
	    info.frontier = beam_search_info.frontier_history[i];
	    info.distance_to_query = distances_from_query_to_all[info.node_index];
	    info.distance_to_query_rank = distances_from_query_to_all_rank[info.node_index];
	    info.timestamp_visit = beam_search_info.visited_timestamp[i];
	    return info;
	  });

	  QueryTraversalFrontierHistoryInfo<unsigned int, Point> traversal_res = {
	    .query_index = query_index,
	    .visited_nodes = visited_node_info,
	    .num_dist_cmps = num_dist_cmps
	  };
	  return traversal_res;
	});
	write_query_traversal_frontier_history_info_to_file(traversal_info, outputFile);
      }
      
    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}
