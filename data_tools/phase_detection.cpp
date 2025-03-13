/*
  This file parses a prebuilt graph file and run beam search queries on this
  graph to get data on the number of hops and time spent in
  each phase of beam search traversal. This data is written to a csv file
  

*/


#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <cmath>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/euclidian_point.h"
#include "utils/beamSearch.h"
#include "utils/csvfile.h"
#include "utils/types.h"


using namespace parlayANN;


void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}

int segmented_least_squares(
			    const parlay::sequence<size_t>& x, 
			    const parlay::sequence<double>& y, 
			    int num_segments = 2) 
{
  // Validate inputs
  if (x.size() != y.size()) abort_with_message("x and y must have equal size");
  if (x.size() == 0) abort_with_message("x and y must not be empty");
    
  const int n = x.size();
    
  
  if (num_segments <= 1 || n < 3) {
    return 0;
  }
  if (!parlay::is_sorted(x)) abort_with_message("x is not sorted");
    
    
    // Function to calculate the error when fitting a line to points from indices i to j (inclusive)
    auto segment_cost = [&](int i, int j) -> double {
        if (j - i < 1) return std::numeric_limits<double>::infinity();
        
        // Calculate means
        double sum_x = 0.0, sum_y = 0.0;
        for (int k = i; k <= j; ++k) {
            sum_x += x[k];
            sum_y += y[k];
        }
        double mean_x = sum_x / (j - i + 1);
        double mean_y = sum_y / (j - i + 1);
        
        // Calculate slope and intercept
        double numerator = 0.0, denominator = 0.0;
        for (int k = i; k <= j; ++k) {
            numerator += (x[k] - mean_x) * (y[k] - mean_y);
            denominator += (x[k] - mean_x) * (x[k] - mean_x);
        }
        
        double slope = (denominator != 0) ? numerator / denominator : 0;
        double intercept = mean_y - slope * mean_x;
        
        // Calculate error
        double error = 0.0;
        for (int k = i; k <= j; ++k) {
            double predicted = slope * x[k] + intercept;
            double diff = y[k] - predicted;
            error += diff * diff;
        }
        
        return error;
    };
    
    // Initialize dynamic programming tables
    // dp[i][j] = min cost to fit j segments on points 0...i
    std::vector<std::vector<double>> dp(n, std::vector<double>(num_segments + 1, std::numeric_limits<double>::infinity()));
    // parent[i][j] = last breakpoint when fitting j segments on points 0...i
    std::vector<std::vector<int>> parent(n, std::vector<int>(num_segments + 1, -1));
    
    // Base case: one segment
    for (int i = 0; i < n; ++i) {
        dp[i][1] = segment_cost(0, i);
    }
    
    // Fill the DP table
    for (int j = 2; j <= num_segments; ++j) {  // For each number of segments
        for (int i = j-1; i < n; ++i) {  // Need at least j points for j segments
            for (int k = j-2; k < i; ++k) {  // Try each possible previous breakpoint
                double cost = dp[k][j-1] + segment_cost(k+1, i);
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    parent[i][j] = k;
                }
            }
        }
    }
    
    // Reconstruct the solution (find breakpoints)
    std::vector<int> breakpoint_indices;
    int i = n - 1;
    int j = num_segments;
    
    while (j > 1) {
        int k = parent[i][j];
        breakpoint_indices.push_back(k + 1);  // +1 because this is the start of the next segment
        i = k;
        j--;
    }
    
    // Reverse since we built the list backwards
    std::reverse(breakpoint_indices.begin(), breakpoint_indices.end());
    
    // Convert indices to x values
    std::vector<double> breakpoints;
    for (int idx : breakpoint_indices) {
        breakpoints.push_back(x[idx]);
    }
    
    return breakpoints[0];
}



/*
  a metric to measure time for phase 2 is defined as the time for beamsearch to converge starting from the nearest neighbor.
 */
template<typename Point, typename PointRange>
double nn_beamsearch_convergence_time(const Point &query_point,
		    const PointRange &Points,
		    const Graph<unsigned int> &G,
		    const groundTruth<unsigned int> &GT,
		    const QueryParams &QP) {
  const parlay::sequence<unsigned int> starting_points = {get_nearest_neighbor(query_point, GT)};
  auto r = beam_search_timestamp(query_point, G, Points,
                                            starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited_timestamp = visited_and_timestamp.second;
  return visited_timestamp[visited_timestamp.size() - 1] - visited_timestamp[0];
}



template <typename Point, typename PointRange>
void record_hops_info(long array_index,
		      const Point &query_point,
                      const Graph<unsigned int> &G,
		      const PointRange &Points,
		      const groundTruth<unsigned int> &GT,
                      const QueryParams &QP, parlay::sequence<int> &num_hops,
                      parlay::sequence<int> &num_hops_phase_1,
                      parlay::sequence<int> &num_hops_phase_2,
                      parlay::sequence<double> &time_phase_1,
                      parlay::sequence<double> &time_phase_2,
                      parlay::sequence<double> &time_total,
		      parlay::sequence<double> &nn_beamsearch_time,
		      parlay::sequence<int> &node_freq, // frequency of each node to show up in search
		      parlay::sequence<int> &phase_1_freq, // frequency of each node to show up in phase 1
		      parlay::sequence<int> &phase_2_freq, // frequency of each node to show up in phase
		      int top_n,
		      bool use_sls = true) {
  const parlay::sequence<unsigned int> starting_points = {0};
  auto r = beam_search_timestamp(query_point, G, Points,
                                            starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited = visited_and_timestamp.first;
  parlay::sequence<double> visited_timestamp = visited_and_timestamp.second;

  parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(
									 Points.size(), [&](long i) { return query_point.distance(Points[i]); });
  parlay::sequence<size_t> distances_from_query_to_all_rank =
    parlay::rank(distances_from_query_to_all);

  parlay::sequence<size_t> distance_visited_rank =
    parlay::tabulate(visited.size(), [&](long i) {
      return distances_from_query_to_all_rank[visited[i].first];
    });

  int phase_1_hops;
  int phase_2_hops;

  double phase_1_time;
  double phase_2_time;
  int breakpoint = 0;
  if (use_sls) {
    const auto log1p_rank = parlay::tabulate(
					     visited.size(),
					     [&](size_t i) {
					       return std::log1p(distance_visited_rank[i]);});
    const auto num_hops = parlay::tabulate(visited.size(), [&](size_t i){return i;});
    breakpoint = segmented_least_squares(num_hops, log1p_rank);
    phase_1_hops = breakpoint;
    phase_2_hops = visited.size() - 1 - phase_1_hops;
    phase_1_time = visited_timestamp[breakpoint] - visited_timestamp[0];
    phase_2_time = visited_timestamp[visited.size() - 1] - visited_timestamp[breakpoint];
  } else {
    ;
    // for (int i = 0; i < visited.size(); i++) {
    //   if (distance_visited_rank[i] <= top_n) {
    // 	phase_1_time =
    //       visited_timestamp[i - 1 >= 0 ? i - 1 : 0] - visited_timestamp[0];
    // 	phase_2_time = visited_timestamp[visited.size() - 1] - visited_timestamp[i];
    // 	phase_1_hops = i;
    // 	phase_2_hops = visited.size() - 1 - phase_1_hops;
    // 	break;
    //   }
    // }
  } 
  num_hops_phase_1[array_index] = phase_1_hops;
  num_hops_phase_2[array_index] = phase_2_hops;
  time_phase_1[array_index] = phase_1_time;
  time_phase_2[array_index] = phase_2_time;

  nn_beamsearch_time[array_index] = nn_beamsearch_convergence_time(
								   query_point,
								   Points,
								   G,
								   GT,
								   QP);
    
  }
  num_hops[array_index] = visited.size();
  time_total[array_index] =
    visited_timestamp[visited.size() - 1] - visited_timestamp[0];
  for (int i = 0; i < visited.size(); i++) {
    node_freq[visited[i].first]++;
    if (i <= breakpoint) {
      phase_1_freq[visited[i].first]++;
    } else {
      phase_2_freq[visited[i].first]++;
    }
  }
}



// this function should be executed sequentially
void record_information_query(
			      long query_index,
			      const parlayANN::PointRange<parlayANN::Euclidian_Point<float>> &Points,
			      const parlayANN::Graph<unsigned int> &G,
			      const parlayANN::QueryParams &QP,
			      const parlay::sequence<unsigned int> starting_points,
			      int top_n, // defines the barrier that separates the 2 phases, nodes that are with the top 100 closest distance to the query node are in phase 2
			      parlay::sequence<double> &total_time,
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
  total_time[query_index] = visited_timestamp[visited.size() - 1] - visited_timestamp[0];
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
                  parlay::sequence<double> &total_time,
		  parlay::sequence<double> &time_phase_1, // time spent in phase 1
		  parlay::sequence<double> &time_phase_2, // time spent in phase 2
		  parlay::sequence<double> &nn_beamsearch_time, // time spent in phase 2
		  parlay::sequence<int> &num_hops_total, // number of hops in phase 1
		  parlay::sequence<int> &num_hops_phase_1, // number of hops in phase 1
		  parlay::sequence<int> &num_hops_phase_2, // number of hops in phase 2, phase 2 includes nodes that are in the top k closest distance to the query node
		  parlay::sequence<int> &node_freq, // frequency of each node to show up in search
		  parlay::sequence<int> &phase_1_freq, // frequency of each node to show up in phase 1
		  parlay::sequence<int> &phase_2_freq // frequency of each node to show up in phase 
		  ) {
  parlayANN::csvfile csv(csv_filename);

  csv << "point_index"
  << "query_time_total"
  << "query_time_phase_1"
  << "query_time_phase_2"
  << "query_num_hops_phase_1"
  << "query_num_hops_phase_2"
  << "node_freq"
  << "phase_1_freq"
  << "phase_2_freq" << parlayANN::endrow;
  
  for (long i = 0; i < time_phase_1.size(); i++) {
    csv << i << total_time[i] << time_phase_1[i] << time_phase_2[i] << num_hops_phase_1[i] << num_hops_phase_2[i]
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
  char* qFile = P.getOptionValue("-query_path");
  char* gtFile = P.getOptionValue("-gt_path");
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

  groundTruth<unsigned int> GT(gtFile);
  
  if (vectype == "float") {
    if (dist_func == "Euclidian"){
      parlayANN::PointRange<parlayANN::Euclidian_Point<float>> Points(iFile);
      parlayANN::PointRange<parlayANN::Euclidian_Point<float>> QueryPoints(qFile);
      if (normalize) abort_with_message("will impl normalization later");
      
      parlayANN::Graph<unsigned int> G(gFile);

      if (quantize == 8 || quantize == 16) abort_with_message( "Will impl support for quantize later");

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

      parlay::sequence<double> time_total(QueryPoints.size(), 0.0);
      parlay::sequence<double> time_phase_1(QueryPoints.size(), 0.0); // time spent in phase 1
      parlay::sequence<double> time_phase_2(QueryPoints.size(), 0.0); // time spent in phase 2
      parlay::sequence<double> nn_beamsearch_time(QueryPoints.size(), 0.0);
      parlay::sequence<int> num_hops_total(QueryPoints.size(), 0);
      parlay::sequence<int> num_hops_phase_1(QueryPoints.size(), 0); // number of hops in phase 1
      parlay::sequence<int> num_hops_phase_2(QueryPoints.size(), 0); // number of hops in phase 2, phase 2 includes nodes that are in the top k closest distance to the query node
      parlay::sequence<int> node_freq(Points.size(), 0); // frequency of each node to show up in search
      parlay::sequence<int> phase_1_freq(Points.size(), 0); // frequency of each node to show up in phase 1
      parlay::sequence<int> phase_2_freq(Points.size(), 0); // frequency of each node to show up in phase 2

      parlay::parallel_for(0, QueryPoints.size(), [&] (long i){
	const Point query_point = QueryPoints[i];
	return record_hops_info(i,
				query_point,
				G,
				Points,
				GT,
				QP, num_hops_total,
				num_hops_phase_1,
				num_hops_phase_2,
				time_phase_1,
				time_phase_2,
				time_total,
				nn_beamsearch_time,
				node_freq,
				phase_1_freq,
				phase_2_freq
				10,
				true
				);
      });
	
      write_to_csv(csv_filename, time_total, time_phase_1, time_phase_2, nn_beamsearch_time, num_hops_total, num_hops_phase_1, num_hops_phase_2, node_freq, phase_1_freq, phase_2_freq);
	
      std::cout << "finished" << std::endl;


      
    } else abort_with_message("Other distance functions are not supported at this moment");
  } else abort_with_message("Other vector types are not supported at this moment");

}

