// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>

#include <algorithm>
#include <queue>
#include <random>
#include <set>

#include "../utils/graph.h"
#include "clusterEdge.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {

struct DisjointSet {
  parlay::sequence<int> parent;
  parlay::sequence<int> rank;
  size_t N;

  DisjointSet(size_t size) {
    N = size;
    parent = parlay::sequence<int>(N);
    rank = parlay::sequence<int>(N);
    parlay::parallel_for(0, N, [&](size_t i) {
      parent[i] = i;
      rank[i] = 0;
    });
  }

  void _union(int x, int y) {
    int xroot = parent[x];
    int yroot = parent[y];
    int xrank = rank[x];
    int yrank = rank[y];
    if (xroot == yroot)
      return;
    else if (xrank < yrank)
      parent[xroot] = yroot;
    else {
      parent[yroot] = xroot;
      if (xrank == yrank) rank[xroot] = rank[xroot] + 1;
    }
  }

  int find(int x) {
    if (parent[x] == x) return x;
    int c = x;
    while (parent[c] != c) {
      c = parent[c];
    }
    while (x != c) {
      int s = parent[x];
      parent[x] = c;
      x = s;
    }
    return c;
  }

  void flatten() {
    for (int i = 0; i < N; i++) find(i);
  }

  bool is_full() {
    flatten();
    parlay::sequence<bool> truthvals(N);
    parlay::parallel_for(
        0, N, [&](size_t i) { truthvals[i] = (parent[i] == parent[0]); });
    auto ff = [&](bool a) { return not a; };
    auto filtered = parlay::filter(truthvals, ff);
    if (filtered.size() == 0) return true;
    return false;
  }
};

template <typename Point, typename PointRange, typename indexType>
struct hcnng_index {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using pid = std::pair<indexType, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  static constexpr indexType kNullId = std::numeric_limits<indexType>::max();
  static constexpr distanceType kNullDist =
      std::numeric_limits<distanceType>::max();
  static constexpr labelled_edge kNullEdge = {{kNullId, kNullId}, kNullDist};

  hcnng_index() {}

  static void remove_edge_duplicates(indexType p, GraphI &G) {
    parlay::sequence<indexType> points;
    for (indexType i = 0; i < G[p].size(); i++) {
      points.push_back(G[p][i]);
    }
    auto np = parlay::remove_duplicates(points);
    G[p].update_neighbors(np);
  }

  void remove_all_duplicates(GraphI &G) {
    parlay::parallel_for(0, G.size(),
                         [&](size_t i) { remove_edge_duplicates(i, G); });
  }

  // inserts each edge after checking for duplicates
  static void process_edges(GraphI &G, parlay::sequence<edge> edges) {
    long maxDeg = G.max_degree();
    auto grouped = parlay::group_by_key(edges);
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      int32_t index = grouped[i].first;
      for (auto c : grouped[i].second) {
        if (G[index].size() < maxDeg) {
          G[index].append_neighbor(c);
        } else {
          remove_edge_duplicates(index, G);
          G[index].append_neighbor(c);
        }
      }
    });
  }

  // parameters dim and K are just to interface with the cluster tree code
  static void MSTk(GraphI &G, PR &Points,
                   parlay::sequence<size_t> &active_indices, long MSTDeg) {
    // preprocessing for Kruskal's
    size_t N = active_indices.size();
    long dim = Points.dimension();
    DisjointSet disjset(N);
    size_t m = 10;
    auto less = [&](labelled_edge a, labelled_edge b) {
      return a.second < b.second;
    };
    parlay::sequence<labelled_edge> candidate_edges(N * m, kNullEdge);
    parlay::parallel_for(0, N, [&](size_t i) {
      std::priority_queue<labelled_edge, std::vector<labelled_edge>,
                          decltype(less)>
          Q(less);
      for (indexType j = i + 1; j < N; j++) {
        distanceType dist_ij =
            Points[active_indices[i]].distance(Points[active_indices[j]]);
        if (Q.size() >= m) {
          distanceType topdist = Q.top().second;
          if (dist_ij < topdist) {
            labelled_edge e;
            e = std::make_pair(std::make_pair(i, j), dist_ij);
            Q.pop();
            Q.push(e);
          }
        } else {
          labelled_edge e;
          e = std::make_pair(std::make_pair(i, j), dist_ij);
          Q.push(e);
        }
      }
      indexType limit = std::min(Q.size(), m);
      for (indexType j = 0; j < limit; j++) {
        candidate_edges[i * m + j] = Q.top();
        Q.pop();
      }
    });

    parlay::sort_inplace(candidate_edges, less);

    auto degrees =
        parlay::tabulate(active_indices.size(), [&](size_t i) { return 0; });
    parlay::sequence<edge> MST_edges = parlay::sequence<edge>();
    // modified Kruskal's algorithm
    for (indexType i = 0; i < candidate_edges.size(); i++) {
      // Since we sorted, any null edges form the suffix.
      if (candidate_edges[i].second == kNullDist) break;
      labelled_edge e_l = candidate_edges[i];
      edge e = e_l.first;
      if ((disjset.find(e.first) != disjset.find(e.second)) &&
          (degrees[e.first] < MSTDeg) && (degrees[e.second] < MSTDeg)) {
        MST_edges.push_back(
            std::make_pair(active_indices[e.first], active_indices[e.second]));
        MST_edges.push_back(
            std::make_pair(active_indices[e.second], active_indices[e.first]));
        degrees[e.first] += 1;
        degrees[e.second] += 1;
        disjset._union(e.first, e.second);
      }
      if (i % N == 0) {
        if (disjset.is_full()) {
          break;
        }
      }
    }
    process_edges(G, std::move(MST_edges));
  }

  // robustPrune routine as found in DiskANN paper, with the exception that the
  // new candidate set is added to the field new_nbhs instead of directly
  // replacing the out_nbh of p
  void robustPrune(indexType p, PR &Points, GraphI &G, double alpha) {
    // add out neighbors of p to the candidate set.
    parlay::sequence<pid> candidates;
    for (size_t i = 0; i < G[p].size(); i++) {
      candidates.push_back(
          std::make_pair(G[p][i], Points[p].distance(Points[G[p][i]])));
    }

    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    parlay::sort_inplace(candidates, less);

    parlay::sequence<int> new_nbhs = parlay::sequence<int>();

    size_t candidate_idx = 0;
    while (new_nbhs.size() < G.max_degree() &&
           candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      indexType p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == kNullId) continue;

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        indexType p_prime = candidates[i].first;
        if (p_prime != kNullId) {
          distanceType dist_starprime =
              Points[p_star].distance(Points[p_prime]);
          distanceType dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime)
            candidates[i].first = kNullId;
        }
      }
    }
    G[p].update_neighbors(new_nbhs);
  }

  void build_index(GraphI &G, PR &Points, long cluster_rounds,
                   long cluster_size, long MSTDeg) {
    cluster<Point, PointRange, indexType> C;
    C.multiple_clustertrees(G, Points, cluster_size, cluster_rounds, MSTk,
                            MSTDeg);
    remove_all_duplicates(G);
    // TODO: enable optional pruning (what is below now works, but
    // should be connected cleanly)
    // parlay::parallel_for(0, G.size(), [&] (size_t i){robustPrune(i, Points,
    // G, 1.1);});
  }
};

}  // namespace parlayANN
