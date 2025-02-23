#!/bin/bash

# Prompt the user for inputs
read -p "Enter the -base_path or the path to the points (default: ../data/sift/sift_learn.fbin): " base_path
base_path=${base_path:-"../data/sift/sift_learn.fbin"}
read -p "Enter the -graph_path or the path to the created graph (default is the hcnng graph: ../data/sift/sift_learn_3_10): " graph_path
graph_path=${graph_path:-"../data/sift/sift_learn_3_10"}
read -p "Enter the vector -data_type (default: float): " data_type
data_type=${data_type:-"float"}
read -p "Enter the vector -dist_func (default: Euclidian): " dist_func
dist_func=${dist_func:-"Euclidian"}
read -p "Enter csv output filename (default: phase_detection_result.csv): " res_filename
res_filename=${res_filename:-"phase_detection_result.csv"}
read -p "Enter png output filename (default: phase_detection_graph.png): " png_filename
png_filename=${png_filename:-"phase_detection_graph.png"}

cd ../data_tools

if [ -f "$res_filename" ]; then
    echo "File '$res_filename' exists. Deleting..."
    rm "$res_filename"
    echo "File deleted."
else
    echo "File '$res_filename' does not exist, will be created."
fi


# Run the program with the provided arguments
./phase_detection -base_path "$base_path" -graph_path "$graph_path" -data_type "$data_type" -dist_func "$dist_func" -res_filename "$res_filename"

cd ../misc_python_functions

python3 generate_phase_detection_graph.py -ip "../data_tools/$res_filename" -op "$png_filename"
