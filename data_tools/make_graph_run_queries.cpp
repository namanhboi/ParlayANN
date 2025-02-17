/*
  This program creates a vamana graph, and doesn't store it, and runs queries on
  it to get the summary statistics about number of hops in phases 1, 2, and in
  the whole graph.

*/
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/csvfile.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/types.h"

#include "../algorithms/vamana/neighbors.h"

using namespace parlayANN;

struct PhasesStats {
  double hops_phase_1_mean;
  double hops_phase_2_mean;
  double hops_total_mean;
  double time_phase_1_mean;
  double time_phase_2_mean;
  double time_total_mean;
};

void abort_with_message(std::string message) {
  std::cout << message << std::endl;
  std::abort();
}

std::vector<int> random_numbers(const int start, const int end, const int num) {
  std::random_device rd;
  std::mt19937 gen(1); // seed
  std::uniform_int_distribution<> distr(start, end);
  std::set<int> rnd_num;
  while (rnd_num.size() < num) {
    rnd_num.insert(distr(gen));
  }
  std::vector<int> v(rnd_num.begin(), rnd_num.end());
  return v;
}

template <typename PointRange>
void record_hops_info(long array_index, long query_index,
                      const Graph<unsigned int> &G, const PointRange &Points,
                      const QueryParams &QP, parlay::sequence<int> &num_hops,
                      parlay::sequence<int> &num_hops_phase_1,
                      parlay::sequence<int> &num_hops_phase_2,
                      parlay::sequence<double> &time_phase_1,
                      parlay::sequence<double> &time_phase_2,
                      parlay::sequence<double> &time_total, int top_n) {

  parlay::sequence<float> distances_from_query_to_all =
      parlay::tabulate(Points.size(), [&](long i) {
        return Points[query_index].distance(Points[i]);
      });
  parlay::sequence<size_t> distances_from_query_to_all_rank =
      parlay::rank(distances_from_query_to_all);

  const parlay::sequence<unsigned int> starting_points = {0};
  auto r = parlayANN::beam_search_timestamp(Points[query_index], G, Points,
                                            starting_points, QP);
  auto visited_and_timestamp = r.first.second;
  auto visited = visited_and_timestamp.first;
  parlay::sequence<double> visited_timestamp = visited_and_timestamp.second;

  parlay::sequence<size_t> distance_visted_rank =
      parlay::tabulate(visited.size(), [&](long i) {
        return distances_from_query_to_all_rank[visited[i].first];
      });

  int phase_1_hops;
  int phase_2_hops;

  double phase_1;
  double phase_2;
  double total;

  for (int i = 0; i < visited.size(); i++) {
    if (distance_visted_rank[i] <= top_n) {
      phase_1 =
          visited_timestamp[i - 1 >= 0 ? i - 1 : 0] - visited_timestamp[0];
      phase_2 = visited_timestamp[visited.size() - 1] - visited_timestamp[i];
      phase_1_hops = i;
      phase_2_hops = visited.size() - phase_1_hops;
      break;
    }
  }
  num_hops_phase_1[array_index] = phase_1_hops;
  num_hops_phase_2[array_index] = phase_2_hops;
  num_hops[array_index] = visited.size();
  time_total[array_index] =
      visited_timestamp[visited.size() - 1] - visited_timestamp[0];
  time_phase_1[array_index] = phase_1;
  time_phase_2[array_index] = phase_2;
}

template <typename PointRange>
PhasesStats calculate_hops_phases(const Graph<unsigned int> &G,
                                  const PointRange &Points,
                                  const int num_query_points,
                                  const QueryParams &QP, const int top_n) {
  std::vector<int> random_indices =
      random_numbers(0, Points.size() - 1, num_query_points);
  parlay::sequence<int> num_hops_phase_1(num_query_points, 0);
  parlay::sequence<int> num_hops_phase_2(num_query_points, 0);
  parlay::sequence<int> num_hops(num_query_points, 0);

  parlay::sequence<double> time_phase_1(num_query_points, 0.0);
  parlay::sequence<double> time_phase_2(num_query_points, 0.0);
  parlay::sequence<double> time_total(num_query_points, 0.0);

  parlay::parallel_for(0, num_query_points, [&](long i) {
    return record_hops_info(i, random_indices[i], G, Points, QP, num_hops,
                            num_hops_phase_1, num_hops_phase_2, time_phase_1,
                            time_phase_2, time_total, top_n);
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
  };
  return stat;
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
  char *vectype = P.getOptionValue("-data_type");
  long num_query_points = P.getOptionIntValue("-num_query", 1000);

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
  if (k > 1000 || k < 0)
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
  groundTruth<unsigned int> GT(NULL);
  if (tp == "float") {
    if (df == "Euclidian") {
      PointRange<Euclidian_Point<float>> Points(iFile);
      PointRange<Euclidian_Point<float>> QueryPoints(NULL);
      if (normalize) {
        for (int i = 0; i < Points.size(); i++)
          Points[i].normalize();
      }
      Graph<unsigned int> G = Graph<unsigned int>(maxDeg, Points.size());
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
        ANN<Point, PR, unsigned int>(G, k, BP, QueryPoints, GT, NULL, false,
                                     Points);
        QueryParams QP;
        QP.limit = limit != -1 ? limit : (long)G.size();
        QP.rerank_factor = rerank_factor;
        QP.degree_limit =
            degree_limit != -1 ? degree_limit : (long)G.max_degree();
        QP.k = k;
        QP.cut = cut;
        QP.beamSize = beam_size;
        PhasesStats stat =
            calculate_hops_phases<PR>(G, Points, num_query_points, QP, top_n);
        std::cout << stat.hops_total_mean << "," << stat.hops_phase_1_mean << ","
        << stat.hops_phase_2_mean << "," << stat.time_total_mean << "," << stat.time_phase_1_mean <<"," << stat.time_phase_2_mean << std::endl;
      }
    }
  } else {
    abort_with_message("currently no support for vectype other than floats");
  }
  return 0;
}
