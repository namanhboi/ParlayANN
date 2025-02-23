#include "phase_utils.h"

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
  PhaseIndices res = {.phase_1_indices= phase_1_indices,
                       .phase_2_indices= phase_2_indices};
  return res;
}



/**
 * create a new graph where the neighborhoods in replacement_indices
 * of graph B is replaced with the corresponding neighborhoods of GraphA , 
*/
parlayANN::Graph<unsigned int>
combine_graphs(parlayANN::Graph<unsigned int> &graph_A, parlayANN::Graph<unsigned int> &graph_B,
              parlay::sequence<unsigned int> replacement_indices) {
  if (graph_A.size() != graph_B.size()) {
    std::cout << "graphs must be same size" << std::endl;
    std::abort();
  }
  parlayANN::Graph<unsigned int> graph_res(
				std::max(graph_A.max_degree(), graph_B.max_degree()), graph_B.size());

  for (int i = 0; i < graph_B.size(); i++) {
    graph_res[i] = graph_B[i];
  }
  for (int i = 0; i < replacement_indices.size(); i++) graph_res[i] = graph_A[i];
  return graph_res;
}



