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

parlay::sequence<int>
num_hops_from_origin(const parlayANN::Graph<unsigned int> &G) {
  std::set<unsigned int> visited;
  parlay::sequence<int> distances(G.size(), 0);
  for (int i = 0; i < G.size(); i++)
    distances.push_back(0);
  std::queue<unsigned int> q;

  q.push(0);
  while (!q.empty()) {
    unsigned int currentNode = q.front();
    q.pop();
    visited.insert(currentNode);
    parlayANN::edgeRange<unsigned int> neighbors = G[currentNode];
    for (size_t i = 0; i < neighbors.size(); i++) {
      unsigned int neighbor = neighbors[i];
      if (visited.find(neighbor) == visited.end()) {
        q.push(neighbor);
        distances[neighbor] = distances[currentNode] + 1;
      }
    }
  }

  return distances;
}

PhaseIndices divide_graph_into_phase(const parlayANN::Graph<unsigned int> &G,
                                     const parlay::sequence<int> &distances,
                                     const int boundary) {
  parlay::sequence<unsigned int> phase_1_indices;
  parlay::sequence<unsigned int> phase_2_indices;
  for (unsigned int i = 0; i < G.size(); i++) {
    if (distances[i] <= boundary) {
      phase_1_indices.push_back(i);
    } else {
      phase_2_indices.push_back(i);
    }
  }
  parlay::integer_sort_inplace(phase_1_indices);
  parlay::integer_sort_inplace(phase_2_indices);
  PhaseIndices res = {.phase_1_indices = phase_1_indices,
                      .phase_2_indices = phase_2_indices};
  return res;
}

/**
 * create a new graph where the neighborhoods in replacement_indices
 * of graph B is replaced with the corresponding neighborhoods of GraphA ,
 */
parlayANN::Graph<unsigned int>
combine_graphs(const parlayANN::Graph<unsigned int> &graph_A,
               const parlayANN::Graph<unsigned int> &graph_B,
               const parlay::sequence<unsigned int> replacement_indices) {
  if (graph_A.size() != graph_B.size()) {
    std::cout << "graphs must be same size" << std::endl;
    std::abort();
  }
  parlayANN::Graph<unsigned int> graph_res(
      std::max(graph_A.max_degree(), graph_B.max_degree()), graph_B.size());

  for (int i = 0; i < graph_B.size(); i++) {
    // graph_res[i] = graph_B[i];
    graph_res[i].append_neighbors(graph_B[i]);
  }
  for (int i = 0; i < replacement_indices.size(); i++) {
    graph_res[replacement_indices[i]].clear_neighbors();
    graph_res[replacement_indices[i]].append_neighbors(
        graph_A[replacement_indices[i]]);
  }
  return graph_res;
}

using namespace parlayANN;

int main(int argc, char *argv[]) {
  commandLine P(argc, argv,
                "[-graph_a <graph_a>] [-graph_b <graph_b>] [-graph_res "
                "<graph_res>] [-boundary]");

  std::string data_type = std::string(P.getOptionValue("-data_type"));
  std::string dist_func = std::string(P.getOptionValue("-dist_func"));
  int boundary = P.getOptionIntValue("-boundary", 0);

  char *graph_a = P.getOptionValue("-graph_a");
  char *graph_b = P.getOptionValue("-graph_b");
  char *graph_res = P.getOptionValue("-graph_res");

  if (data_type == "float") {
    if (dist_func == "Euclidian") {
      const Graph<unsigned int> GraphA(graph_a);
      const Graph<unsigned int> GraphB(graph_b);
      const PhaseIndices phases = divide_graph_into_phase(
          GraphA, num_hops_from_origin(GraphA), boundary);
      Graph<unsigned int> res =
          combine_graphs(GraphA, GraphB, phases.phase_1_indices);
      res.save(graph_res);
      for (int i = 0; i < 100; i++) {
        std::cout << "bruh";
        for (int j = 0; j < res[i].size(); j++) {
          std::cout << res[i][j] << ",";
        }
      }
      std::cout << std::endl;
    }
  }
  return 0;
}
