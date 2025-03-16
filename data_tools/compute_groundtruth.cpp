/*
  Example usage:
    ./compute_groundtruth -base_path ~/data/sift/sift-1M \
    -query_path ~/data/sift/query-10K -data_type uint8 \
    -dist_func Euclidian -k 100 -gt_path ~/data/sift/GT/sift-1M.gt
*/

#include <iostream>
#include <algorithm>
#include <cstdint>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "../algorithms/bench/parse_command_line.h"
#include "utils/mmap.h"


using pid = std::pair<int, float>;
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




template<typename PointRange>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth(PointRange &B, 
  PointRange &Q, int k){
    unsigned d = B.dimension();
    size_t q = Q.size();
    size_t b = B.size();
    auto answers = parlay::tabulate(q, [&] (size_t i){  
        float topdist = B[0].d_min();   
        int toppos;
        parlay::sequence<pid> topk;
        for(size_t j=0; j<b; j++){
            // float dist = D->distance((Q[i].coordinates).begin(), (B[j].coordinates).begin(), d);
            float dist = Q[i].distance(B[j]);
            if(topk.size() < k){
                if(dist > topdist){
                    topdist = dist;   
                    toppos = topk.size();
                }
                topk.push_back(std::make_pair((int) j, dist));
            }
            else if(dist < topdist){
                float new_topdist=B[0].d_min();  
                int new_toppos=0;
                topk[toppos] = std::make_pair((int) j, dist);
                for(size_t l=0; l<topk.size(); l++){
                    if(topk[l].second > new_topdist){
                        new_topdist = topk[l].second;
                        new_toppos = (int) l;
                    }
                }
                topdist = new_topdist;
                toppos = new_toppos;
            }
        }
        return topk;
    });
    std::cout << "Done computing groundtruth" << std::endl;
    return answers;
}


// ranking format:
// preamble: number of querypoints, number of points
void write_ranking(
			const parlay::sequence<parlay::sequence<int>> &results,
			const std::string outFile){
  std::cout << "Writing file with number of points:  " << results[0].size() << std::endl;
  std::cout << "File contains groundtruth for " << results.size() << " query points" << std::endl;

  int num_query_points = static_cast<int>(results.size());
  int num_base_points  = static_cast<int>(results[0].size());
  parlay::sequence<int> preamble = {num_query_points, num_base_points};

  parlay::sequence<int> flat_ranking = parlay::flatten(results);

  auto pr = preamble.begin();
  auto ranking_data = flat_ranking.begin();

  std::ofstream writer;
  writer.open(outFile, std::ios::binary | std::ios::out);
  writer.write((char *) pr, 2*sizeof(int));
  writer.write((char *) ranking_data, num_query_points * num_base_points * sizeof(int));
  writer.close();
}


// ibin is the same as the binary groundtruth format used in the
// big-ann-benchmarks (see: https://big-ann-benchmarks.com/neurips21.html)
void write_ibin(parlay::sequence<parlay::sequence<pid>> &result, const std::string outFile, int k){
    std::cout << "Writing file with dimension " << result[0].size() << std::endl;
    std::cout << "File contains groundtruth for " << result.size() << " query points" << std::endl;

    auto less = [&] (pid a, pid b) {return a.second < b.second;};
    parlay::sequence<int> preamble = {static_cast<int>(result.size()), static_cast<int>(result[0].size())};
    size_t n = result.size();
    parlay::parallel_for(0, result.size(), [&] (size_t i){
      parlay::sort_inplace(result[i], less);
    });
    auto ids = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<int> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<int>(result[i][j].first));
        }
        return data;
    });
    auto distances = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<float> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<float>(result[i][j].second));
        }
        return data;
    });
    parlay::sequence<int> flat_ids = parlay::flatten(ids);
    parlay::sequence<float> flat_dists = parlay::flatten(distances);

    auto pr = preamble.begin();
    auto id_data = flat_ids.begin();
    auto dist_data = flat_dists.begin();
    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) pr, 2*sizeof(int));
    writer.write((char *) id_data, n * k * sizeof(int));
    writer.write((char *) dist_data, n * k * sizeof(float));
    writer.close();
}


bool check_groundtruth_ranking(
			       const parlay::sequence<parlay::sequence<int>> &calculated_result,
			       const char* gtFile) {
  auto [fileptr, length] = mmapStringFromFile(gtFile);
  int num_query_vectors = *((int*) fileptr);
  int num_base_vectors = *((int*) (fileptr + sizeof(int)));

  int* start_ranking =  (int*) (fileptr + 2 * sizeof(int));
  int* end_ranking = start_ranking + num_query_vectors * num_base_vectors;
  
  const auto file_result = parlay::slice(start_ranking, end_ranking);
  const auto flat_calculated_result = parlay::flatten(calculated_result);
  const auto slice_result = parlay::slice(
					  flat_calculated_result.begin(),
					  flat_calculated_result.end());
  std::cout << num_query_vectors << " " << calculated_result.size() << std::endl;
  std::cout << num_base_vectors << " " << calculated_result[0].size() << std::endl;
  for (int i = 0; i < 10; i++) {
     std::cout << slice_result[i] << ' ' << file_result[i] << std::endl;
  }
  return parlay::equal(slice_result, file_result);
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* gFile = P.getOptionValue("-gt_path");
  char* qFile = P.getOptionValue("-query_path");
  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  int k = P.getOptionIntValue("-k", 100);
  bool ranking = P.getOption("-ranking");

  std::string df = std::string(dfc);
  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: invalid distance type: specify Euclidian or mips" << std::endl;
    abort();
  }

  std::string tp = std::string(vectype);
  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: data type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  std::cout << "Computing the " << k << " nearest neighbors" << std::endl;

  int maxDeg = 0;

  parlay::sequence<parlay::sequence<pid>> answers;
  std::string base = std::string(bFile);
  std::string query = std::string(qFile);

  if (ranking) {
    if (tp == "float") {
      if (df == "Euclidian") {
	auto Points = PointRange<Euclidian_Point<float>>(bFile);
	auto QueryPoints = PointRange<Euclidian_Point<float>>(qFile);
	const parlay::sequence<unsigned int> starting_points = {0};
	parlay::sequence<parlay::sequence<int>> rankings = compute_groundtruth_rank(
										    Points,
										    QueryPoints,
										    starting_points);
	write_ranking(rankings, std::string(gFile));
	std:: cout << check_groundtruth_ranking(
				  rankings,
				  gFile
				  );
      }
    }
  } else {
    if(tp == "float"){
      std::cout << "Detected float coordinates" << std::endl;
      if(df == "Euclidian"){
	auto B = PointRange<Euclidian_Point<float>>(bFile);
	auto Q = PointRange<Euclidian_Point<float>>(qFile);
	answers = compute_groundtruth<PointRange<Euclidian_Point<float>>>(B, Q, k);
      } else if(df == "mips"){
	auto B = PointRange<Mips_Point<float>>(bFile);
	auto Q = PointRange<Mips_Point<float>>(qFile);
	answers = compute_groundtruth<PointRange<Mips_Point<float>>>(B, Q, k);
      }
    }else if(tp == "uint8"){
      std::cout << "Detected uint8 coordinates" << std::endl;
      if(df == "Euclidian"){
	auto B = PointRange<Euclidian_Point<uint8_t>>(bFile);
	auto Q = PointRange<Euclidian_Point<uint8_t>>(qFile);
	answers = compute_groundtruth<PointRange<Euclidian_Point<uint8_t>>>(B, Q, k);
      } else if(df == "mips"){
	auto B = PointRange<Mips_Point<uint8_t>>(bFile);
	auto Q = PointRange<Mips_Point<uint8_t>>(qFile);
	answers = compute_groundtruth<PointRange<Mips_Point<uint8_t>>>(B, Q, k);
      }
    } else if(tp == "int8"){
      std::cout << "Detected int8 coordinates" << std::endl;
      if(df == "Euclidian"){
	auto B = PointRange<Euclidian_Point<int8_t>>(bFile);
	auto Q = PointRange<Euclidian_Point<int8_t>>(qFile);
	answers = compute_groundtruth<PointRange<Euclidian_Point<int8_t>>>(B, Q, k);
      } else if(df == "mips"){
	auto B = PointRange<Mips_Point<int8_t>>(bFile);
	auto Q = PointRange<Mips_Point<int8_t>>(qFile);
	answers = compute_groundtruth<PointRange<Mips_Point<int8_t>>>(B, Q, k);
      }
    }
    write_ibin(answers, std::string(gFile), k);
  }
  return 0;
}
