/*
  This program creates a vamana graph, and doesn't store it, and runs queries on
  it to get the summary statistics about number of hops in phases 1, 2, and in
  the whole graph.

  This is used for optuna training
*/
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <cmath>

#include "../algorithms/bench/parse_command_line.h"
#include "../algorithms/vamana/neighbors.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/csvfile.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/types.h"
#include "utils/csvfile.h"

using namespace parlayANN;

template<typename PointRange>
parlay::sequence<parlay::sequence<int>> compute_groundtruth_rank(
								    const PointRange &Points,
								    const PointRange &QueryPoints,
								    const parlay::sequence<unsigned int> starting_points) {
  const parlay::sequence<parlay::sequence<int>> all_query_ranking =
    parlay::tabulate(QueryPoints.size(), [&](long query_index) {
      const parlay::sequence<float> distances_from_query_to_all = parlay::tabulate(
									     Points.size(),
									     [&](long i) {return QueryPoints[query_index].distance(Points[i]); });
      const parlay::sequence<size_t> distances_from_query_to_all_rank =
	parlay::rank(distances_from_query_to_all);
      return parlay::tabulate(distances_from_query_to_all_rank.size(),
			      [&] (long i) {
				return static_cast<int>(distances_from_query_to_all_rank[i]);
			      });
    });
  
  return all_query_ranking;
}



/*
  The division between phase 1 and 2 is determined by a the breakpoint
  in segmented least squared with 2 segments
  
  time_nn_beamsearch_convergence is the time spent for beamsearch to converge when
  starting at the nearest neighbor of a query node. 

*/
struct PhasesStats {
  double hops_phase_1_mean;
  double hops_phase_2_mean;
  double hops_total_mean;
  double time_phase_1_mean;
  double time_phase_2_mean;
  double time_total_mean;
  double time_nn_beamsearch_convergence_mean;
  parlay::sequence<int> num_hops_phase_1;
  parlay::sequence<int> num_hops_phase_2 ;
  parlay::sequence<int> num_hops;
  parlay::sequence<double> time_phase_1;
  parlay::sequence<double> time_phase_2;
  parlay::sequence<double> time_total;
  parlay::sequence<double> nn_beamsearch_time;
};


enum class OptMetric {
  AVG_TOTAL_TIME,
  AVG_PHASE_1_TIME,
  AVG_PHASE_2_TIME,
  ALL,
  RECALL,
  NN_BEAMSEARCH_CONVERGENCE
};


void print_opt_metric(OptMetric opt_metric){
  switch(opt_metric) {
  case OptMetric::AVG_TOTAL_TIME:
    std::cout<<"avg total time"<<std::endl;
    break;
  case OptMetric::AVG_PHASE_1_TIME:
    std::cout<<"avg phase 1 time"<<std::endl;
    break;
  case OptMetric::AVG_PHASE_2_TIME:
    std::cout<<"avg phase 2 time"<<std::endl;
    break;
  case OptMetric::ALL:
    std::cout<<"all"<<std::endl;
    break;
  case OptMetric::NN_BEAMSEARCH_CONVERGENCE:
    std::cout << "nn beamsearch convergence" << std::endl;
    break;
  }
}

void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}


/**
 * Perform segmented least squares regression and return only the breakpoints.
 * 
 * @param x Vector of x coordinates (must be sorted in ascending order)
 * @param y Vector of y coordinates
 * @param num_segments Number of line segments to fit
 * @return Vector of breakpoint x-coordinates between segments
 */
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





template<typename Point>
unsigned int get_nearest_neighbor(const Point &query_point,
				  const groundTruth<unsigned int> &GT) {
  return GT.coordinates(query_point.id(), 0);
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
  auto r = parlayANN::beam_search_timestamp(query_point, G, Points,
                                            starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited_timestamp = visited_and_timestamp.second;
  return visited_timestamp[visited_timestamp.size() - 1] - visited_timestamp[0];
}

/*
  query_point might not necessarily be in Points.
  returns list of knn and  distance
*/
template<typename Point, typename PointRange>
// parlay::sequence<std::pair<unsigned int, double>>
void print_top_k_distance_neighbors(
				    const Point &query_point,
			       const PointRange &Points,
			       const groundTruth<unsigned int> &GT,
			       const int k) {
  for (int i = 0; i < k; i++) {
    unsigned int actual_index_topk_point = GT.coordinates(query_point.id(), i);
    double distance = query_point.distance(Points[actual_index_topk_point]);
    std::cout << "Top " << i <<  " point index is " << actual_index_topk_point << "," << "Distance from point to top " << i << " point is " << distance << std::endl;
  }
  
}

template<typename Point, typename PointRange>
void print_actual_topk_distance(const Point &query_point,
				const PointRange &Points,
				int k) {
  auto less = [&] (std::pair<double, unsigned int> a,
		   std::pair<double, unsigned int> b) -> bool {
    return a.first < b.first;
  };
  auto distances = parlay::tabulate(Points.size(), [&] (size_t i) -> std::pair<double, unsigned int> {
    return std::make_pair(query_point.distance(Points[i]), i);
  });
  parlay::sort_inplace(distances, less);
  for (int i = 0; i < k; i++ ) {
    std::cout << distances[i].second << " " << distances[i].first << std::endl;
  }

}




template <typename Point, typename PointRange>
void record_hops_info(OptMetric opt_metric, long query_index,
		      const Point &query_point,
                      const Graph<unsigned int> &G,
		      const PointRange &Points,
		      const groundTruth<unsigned int> &GT,
		      const parlay::sequence<parlay::sequence<int>> &GT_ranking,
                      const QueryParams &QP, parlay::sequence<int> &num_hops,
                      parlay::sequence<int> &num_hops_phase_1,
                      parlay::sequence<int> &num_hops_phase_2,
                      parlay::sequence<double> &time_phase_1,
                      parlay::sequence<double> &time_phase_2,
                      parlay::sequence<double> &time_total,
		      parlay::sequence<double> &nn_beamsearch_time,
		      int top_n,
		      bool use_sls = true) {
  const parlay::sequence<unsigned int> starting_points = {0};
  auto r = parlayANN::beam_search_timestamp(query_point, G, Points,
                                            starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited = visited_and_timestamp.first;
  parlay::sequence<double> visited_timestamp = visited_and_timestamp.second;

  if (opt_metric == OptMetric::ALL ||
      opt_metric == OptMetric::AVG_PHASE_2_TIME ||
      opt_metric == OptMetric::AVG_PHASE_1_TIME) {
    int phase_1_hops;
    int phase_2_hops;

    double phase_1_time;
    double phase_2_time;

    if (use_sls) {
      const auto log1p_rank = parlay::tabulate(
					       visited.size(),
					       [&](size_t i) {
						 return std::log1p(GT_ranking[query_index][i]);});
      const auto num_hops = parlay::tabulate(visited.size(), [&](size_t i){return i;});
      int breakpoint = segmented_least_squares(num_hops, log1p_rank);
      phase_1_hops = breakpoint;
      phase_2_hops = visited.size() - 1 - phase_1_hops;
      phase_1_time = visited_timestamp[breakpoint] - visited_timestamp[0];
      phase_2_time = visited_timestamp[visited.size() - 1] - visited_timestamp[breakpoint];
    } else {
      for (int i = 0; i < visited.size(); i++) {
	if (GT_ranking[query_index][i] <= top_n) {
	  phase_1_time =
            visited_timestamp[i - 1 >= 0 ? i - 1 : 0] - visited_timestamp[0];
	  phase_2_time = visited_timestamp[visited.size() - 1] - visited_timestamp[i];
	  phase_1_hops = i;
	  phase_2_hops = visited.size() - 1 - phase_1_hops;
	  break;
	}
      }
    } 
    num_hops_phase_1[query_index] = phase_1_hops;
    num_hops_phase_2[query_index] = phase_2_hops;
    time_phase_1[query_index] = phase_1_time;
    time_phase_2[query_index] = phase_2_time;
  }
  if (opt_metric == OptMetric::ALL || opt_metric == OptMetric::NN_BEAMSEARCH_CONVERGENCE) {
    nn_beamsearch_time[query_index] = nn_beamsearch_convergence_time(
								     query_point,
								     Points,
								     G,
								     GT,
								     QP);
    
  }
  num_hops[query_index] = visited.size();
  time_total[query_index] =
      visited_timestamp[visited.size() - 1] - visited_timestamp[0];
}

template <typename PointRange>
PhasesStats calculate_hops_phases(const OptMetric opt_metric,
				  const Graph<unsigned int> &G,
                                  const PointRange &Points,
                                  const PointRange &QueryPoints,
				  const groundTruth<unsigned int> &GT,
				  const parlay::sequence<parlay::sequence<int>> &GT_ranking,
                                  const QueryParams &QP, const int top_n) {
  const size_t num_query_points = QueryPoints.size();
  parlay::sequence<int> num_hops_phase_1(num_query_points, 0);
  parlay::sequence<int> num_hops_phase_2(num_query_points, 0);
  parlay::sequence<int> num_hops(num_query_points, 0);

  parlay::sequence<double> time_phase_1(num_query_points, 0.0);
  parlay::sequence<double> time_phase_2(num_query_points, 0.0);
  parlay::sequence<double> time_total(num_query_points, 0.0);
  parlay::sequence<double> nn_beamsearch_time(num_query_points, 0.0);
  parlay::parallel_for(0, num_query_points, [&](long i) {
    return record_hops_info(opt_metric,
			    i,
			    QueryPoints[i],
			    G,
			    Points,
			    GT,
			    GT_ranking,
			    QP,
			    num_hops,
                            num_hops_phase_1,
			    num_hops_phase_2,
			    time_phase_1,
                            time_phase_2,
			    time_total,
			    nn_beamsearch_time,
			    top_n);
  });

  PhasesStats stat = {
      .hops_phase_1_mean =
          (double)parlay::reduce(num_hops_phase_1) / num_query_points,
      .hops_phase_2_mean =
          (double)parlay::reduce(num_hops_phase_2) / num_query_points,
      .hops_total_mean = (double)parlay::reduce(num_hops) / num_query_points,
      .time_phase_1_mean = parlay::reduce(time_phase_1) / num_query_points,
      .time_phase_2_mean = parlay::reduce(time_phase_2) / num_query_points,
      .time_total_mean = parlay::reduce(time_total) / num_query_points,
      .time_nn_beamsearch_convergence_mean = parlay::reduce(nn_beamsearch_time) / num_query_points,
      .num_hops_phase_1 = num_hops_phase_1,
      .num_hops_phase_2 = num_hops_phase_2,
      .num_hops = num_hops,
      .time_phase_1 = time_phase_1,
      .time_phase_2 = time_phase_2,
      .time_total = time_total,
      .nn_beamsearch_time = nn_beamsearch_time,
  };
  return stat;
}



void write_to_csv(std::string csv_filename, PhasesStats stat) {
  csvfile csv(csv_filename);

  csv << "point_index"
  << "query_time_total"
  << "query_time_phase_1"
  << "query_time_phase_2"
  << "nn_beamsearch_time"
  << "query_num_hops_total" 
  << "query_num_hops_phase_1"
  << "query_num_hops_phase_2" << endrow;

  
  for (long i = 0; i < stat.time_phase_1.size(); i++) {
    csv << i
    << stat.time_total[i]
    << stat.time_phase_1[i]
    << stat.time_phase_2[i]
    << stat.nn_beamsearch_time[i]
    << stat.num_hops[i]
    << stat.num_hops_phase_1[i]
    << stat.num_hops_phase_2[i] << endrow;
  }
  csv << endrow;
  csv << endrow;
}


int main(int argc, char *argv[]) {
  commandLine P(argc, argv,
                "[-a <alpha>] [-d <delta>] [-R <deg>]"
                "[-L <bm>] [-k <k> ]  [-gt_path <g>] [-query_path <qF>]"
                "[-graph_path <gF>] [-graph_outfile <oF>] [-res_path <rF>]"
                "[-num_passes <np>]"
                "[-memory_flag <algoOpt>] [-mst_deg <q>] [-num_clusters <nc>] "
                "[-cluster_size <cs>]"
                "[-data_type <tp>] [-dist_func <df>] [-base_path <b>] "
                "[-distance_origin <do>]<inFile>");

  char *iFile = P.getOptionValue("-base_path");
  char *graph_file = P.getOptionValue("-graph_path");
  char *rFile =
    P.getOptionValue("-res_path"); // path for result of making ANN index
  char *qFile = P.getOptionValue("-query_path");
  char *gtFile = P.getOptionValue("-gt_path");
  char *gtRankingFile = P.getOptionValue("-gt_rank_path");  
  char *vectype = P.getOptionValue("-data_type");
  char *metric = P.getOptionValue("-metric");
  if (metric == NULL) {
    std::cout << "need to specify a metric" << std::endl;
    std::abort();
  }
  if (strcmp(metric, "avg_phase_1_time") &&
      strcmp(metric, "avg_phase_2_time") &&
      strcmp(metric, "avg_total_time") &&
      strcmp(metric, "all") &&
      strcmp(metric, "recall") &&
      strcmp(metric, "nn_beamsearch")) {
    std::cout << "-metric can only have one of the following values: avg_total_time, avg_phase_1_time, avg_phase_2_time, nn_beamsearch" << std::endl;
    std::abort();
  }

  OptMetric opt_metric;
  if (!strcmp(metric, "avg_phase_1_time")) opt_metric = OptMetric::AVG_PHASE_1_TIME;
  if (!strcmp(metric, "avg_phase_2_time")) opt_metric = OptMetric::AVG_PHASE_2_TIME;
  if (!strcmp(metric, "avg_total_time")) opt_metric = OptMetric::AVG_TOTAL_TIME;
  if (!strcmp(metric, "all")) opt_metric = OptMetric::ALL;
  if (!strcmp(metric, "recall")) opt_metric = OptMetric::RECALL;
  if (!strcmp(metric, "nn_beamsearch")) opt_metric = OptMetric::NN_BEAMSEARCH_CONVERGENCE;
  
  long Q = P.getOptionIntValue("-Q", 0);
  long R = P.getOptionIntValue("-R", 0);
  if (R < 0)
    P.badArgument();
  long L = P.getOptionIntValue("-L", 0);
  if (L < 0)
    P.badArgument();
  long MST_deg = P.getOptionIntValue("-mst_deg", 0);
  if (MST_deg < 0)
    P.badArgument();
  long num_clusters = P.getOptionIntValue("-num_clusters", 0);
  if (num_clusters < 0)
    P.badArgument();
  long cluster_size = P.getOptionIntValue("-cluster_size", 0);
  if (cluster_size < 0)
    P.badArgument();
  double radius = P.getOptionDoubleValue("-radius", 0.0);
  double radius_2 = P.getOptionDoubleValue("-radius_2", radius);
  long k = P.getOptionIntValue("-k", 10);
  if ( k < 0)
    P.badArgument();
  double alpha = P.getOptionDoubleValue("-alpha", 1.0);
  int num_passes = P.getOptionIntValue("-num_passes", 1);
  int two_pass = P.getOptionIntValue("-two_pass", 0);
  if (two_pass > 1 | two_pass < 0)
    P.badArgument();
  if (two_pass == 1)
    num_passes = 2;
  double delta = P.getOptionDoubleValue("-delta", 0);
  if (delta < 0)
    P.badArgument();
  char *dfc = P.getOptionValue("-dist_func");
  int quantize = P.getOptionIntValue("-quantize_bits", 0);
  int quantize_build = P.getOptionIntValue("-quantize_mode", 0);
  bool verbose = P.getOption("-verbose");
  bool normalize = P.getOption("-normalize");
  double trim = P.getOptionDoubleValue("-trim", 0.0); // not used
  bool self = P.getOption("-self");
  int rerank_factor = P.getOptionIntValue("-rerank_factor", 100);
  bool range = P.getOption("-range");
  int distance_origin = P.getOptionIntValue(
      "-distance_origin",
      0); // distance of all other vectors from the specified vector

  // this integer represents the number of random edges to start with for
  // inserting in a single batch per round
  int single_batch = P.getOptionIntValue("-single_batch", 0);

  // Query params
  long limit = P.getOptionLongValue("-limit", -1);
  long degree_limit = P.getOptionLongValue("-dergee_limit", -1);
  int beam_size = P.getOptionIntValue("-beam_size", 10);
  double cut = P.getOptionDoubleValue("-cut", 1.35);
  int top_n = P.getOptionIntValue("-top_n", 100);

  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);

  BuildParams BP =
      BuildParams(R, L, alpha, num_passes, num_clusters, cluster_size, MST_deg,
                  delta, verbose, quantize_build, radius, radius_2, self, range,
                  single_batch, Q, trim, rerank_factor);
  long maxDeg = BP.max_degree();

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  if (df != "Euclidian" && df != "mips") {
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }
  groundTruth<unsigned int> GT(gtFile);

  
  std::cout << GT.size() << std::endl;
  

  if (tp == "float") {
    if (df == "Euclidian") {
      PointRange<Euclidian_Point<float>> Points(iFile);
      PointRange<Euclidian_Point<float>> QueryPoints(qFile);
      PointRange<Euclidian_Point<float>> EmptyPoints(NULL);
            
      
      std::cout << "Number of points: " << Points.size() << std::endl;
      std::cout << "Number of query points: " << QueryPoints.size() << std::endl;
      std::cout << "Number of ground truth points: " << GT.size() << std::endl;
      if (quantize == 8) {
        using QT = uint8_t;
        using QPoint = Euclidian_Point<QT>;
        using PR = PointRange<QPoint>;
        PR Points_(Points);
        // missing impl
        abort_with_message("haven't implemented quantize = 8 yet");
      } else if (quantize == 16) {
        using Point = Euclidian_Point<uint16_t>;
        using PR = PointRange<Point>;
        PR Points_(Points);
        // missing impl
        abort_with_message("haven't implemented quantize = 16 yet");
      } else {
        using Point = Euclidian_Point<float>;
        using PR = PointRange<Point>;
	Point sample_query_point = QueryPoints[0];
	// print_top_k_distance_neighbors<Point, PR>(sample_query_point, Points, GT, k);
	// print_actual_topk_distance<Point, PR> (sample_query_point, Points, k);
	Graph<unsigned int> G(maxDeg, Points.size());
	if (graph_file == NULL) {
	  G = Graph<unsigned int>(maxDeg, Points.size());
	  ANN<Point, PR, unsigned int>(G, k, BP, EmptyPoints, GT, NULL, false,
                                       Points);
	} else {
	  G = Graph<unsigned int>(graph_file);
	}
	parlay::sequence<unsigned int> starting_points{0};
	parlay::sequence<parlay::sequence<int>> GT_ranking;
	if (gtRankingFile != NULL) {
	  std::cout << "Loading instead of computing" << std::endl;
	  auto [fileptr, length] = mmapStringFromFile(gtRankingFile);
	  int num_query_vectors = *((int*) fileptr);
	  int num_base_vectors = *((int*) (fileptr + sizeof(int)));
    
	  int* start_ranking =  (int*) (fileptr + 2 * sizeof(int));
	  int* end_ranking = start_ranking + num_query_vectors * num_base_vectors;
    
	  auto ranking_flat = parlay::slice(start_ranking, end_ranking);
	  for (int i = 0; i < num_query_vectors; i++) {
	    // const auto slice_i = parlay::slice(ranking_flat.begin() + i * num_base_vectors,
								// ranking_flat.begin() + (i + 1) * num_base_vectors);
								GT_ranking.push_back(parlay::to_sequence(parlay::slice(
														       ranking_flat.begin() + i * num_base_vectors,
														       ranking_flat.begin() + (i + 1) * num_base_vectors)));
	  }
	} else {
	  GT_ranking = compute_groundtruth_rank<PR>(Points,
						QueryPoints,
						starting_points);
	}
	
	

	
        QueryParams QP;
        QP.limit = limit != -1 ? limit : (long)G.size();
        QP.rerank_factor = rerank_factor;
        QP.degree_limit =
            degree_limit != -1 ? degree_limit : (long)G.max_degree();
        QP.k = k;
        QP.cut = cut;
        QP.beamSize = beam_size;
	PhasesStats stat = {
	  .hops_phase_1_mean = 0,
	  .hops_phase_2_mean = 0,
	  .hops_total_mean = 0,
	  .time_phase_1_mean = 0,
	  .time_phase_2_mean = 0,
	  .time_total_mean = 0,
	  .time_nn_beamsearch_convergence_mean = 0
	};
	if (opt_metric != OptMetric::RECALL) {
	  print_opt_metric(opt_metric);
	  stat =
            calculate_hops_phases<PR>(
				      opt_metric,
				      G,
				      Points,
				      QueryPoints,
				      GT,
				      GT_ranking,
				      QP,
				      top_n);
	}
        nn_result recall_res = checkRecall<PR, PR, PR, unsigned int>(
								     G,
								     Points,
								     QueryPoints,
								     Points,
								     QueryPoints,
								     Points,
								     QueryPoints,
								     GT,
								     false,
								     0,
								     10,
								     QP,
								     false);
	
	std::cout <<
	stat.hops_total_mean << "," <<
	stat.hops_phase_1_mean << "," <<
	stat.hops_phase_2_mean << "," <<
	stat.time_total_mean << "," <<
	stat.time_phase_1_mean << "," <<
	stat.time_phase_2_mean << "," <<
	recall_res.recall << "," <<
	stat.time_nn_beamsearch_convergence_mean << std::endl;
	if (rFile != NULL) {
	  std::string res_path(rFile);
	  write_to_csv(res_path, stat);
	}
      }
    }
  } else {
    abort_with_message("currently no support for vectype other than floats");
  }
  return 0;
}
