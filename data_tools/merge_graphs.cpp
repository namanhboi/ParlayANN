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
#include "phase_utils.h"


using namespace parlayANN;


int main(int argc, char *argv[]) {
  commandLine P(argc, argv,
                "[-graph_a <graph_a>] [-graph_b <graph_b>] [-graph_res <graph_res>] [-boundary]");

  char *points_file = P.getOptionValue("-base_path");
  std::string data_type= std::string(P.getOptionValue("-data_type"));
  std::string dist_func = std::string(P.getOptionValue("-dist_func"));
  int boundary = P.getOptionIntValue("-boundary", 0);
  
  char *graph_a = P.getOptionValue("-graph_a");
  char *graph_b = P.getOptionValue("-graph_b");
  char *graph_res = P.getOptionValue("-graph_res");
  
  
  if (data_type == "float") {
    if (dist_func == "Euclidian") {
      PointRange<Euclidian_Point<float>> Points(points_file);
      Graph<unsigned int> GraphA(graph_a);
      Graph<unsigned int> GraphB(graph_b);
      PhaseIndices phases = divide_graph_into_phase(GraphA, num_hops_from_origin(GraphA), boundary);
      Graph<unsigned int> res =
        combine_graphs(GraphA, GraphB, phases.phase_1_indices);
      res.save(graph_res);
    }
  }
  return 0;
}
