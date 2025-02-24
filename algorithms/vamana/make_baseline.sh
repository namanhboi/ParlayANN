# bash
NAME=baseline
BUILD_ARGS="-R 77 -L 160 -alpha 1.175 -two_pass 0"
TYPE_ARGS="-data_type float -dist_func Euclidian"

DATA_FILE=../../data/sift/sift_learn.fbin
GRAPH_FILE=../../data/sift/vamana_baseline_77_160_1_175

# build
./neighbors $BUILD_ARGS $TYPE_ARGS -base_path $DATA_FILE -graph_outfile $GRAPH_FILE

