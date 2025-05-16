#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "../algorithms/bench/parse_command_line.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/csvfile.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"

struct PhaseIndices {
  parlay::sequence<unsigned int> phase_1_indices;
  parlay::sequence<unsigned int> phase_2_indices;
};

/**
 * returns the bfs distances from origin.
 */
parlay::sequence<int>
num_hops_from_origin(const parlayANN::Graph<unsigned int> &G);

/**
 * returns the sorted indices for nodes of each phase
 */
PhaseIndices divide_graph_into_phase(const parlayANN::Graph<unsigned int> &G,
                                     const parlay::sequence<int> &distances,
                                     const int boundary);

/**
 * create a new graph where the neighborhoods in replacement_indices
 * of graph B is replaced with the corresponding neighborhoods of GraphA ,
 */
parlayANN::Graph<unsigned int>
combine_graphs(parlayANN::Graph<unsigned int> &graph_A,
               parlayANN::Graph<unsigned int> &graph_B,
               parlay::sequence<unsigned int> replacement_indices);
