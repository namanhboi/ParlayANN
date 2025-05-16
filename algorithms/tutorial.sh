#!/bin/bash

cd vamana
make
echo "Vamana:"
./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin

echo "" 
echo "" 

cd ../HCNNG
make
echo "HCNNG:"
./neighbors -cluster_size 1000 -mst_deg 3 -num_clusters 30  -graph_outfile ../../data/sift/sift_learn_3_10 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/hcnng_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin


echo ""
echo ""

cd ../pyNNDescent
make
echo "pyNNDescent:"
./neighbors -R 40 -cluster_size 100 -num_clusters 10 -alpha 1.2 -delta 0.05 -graph_outfile ../../data/sift/sift_learn_30 -query_path ../../data/sift/sift_query.fbin -gt_path ../../data/sift/sift-100K -res_path ../../data/pynn_res.csv -data_type float -dist_func Euclidian -base_path ../../data/sift/sift_learn.fbin

