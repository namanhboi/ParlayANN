import argparse
import subprocess
import os

vamana_binary_path = "../algorithms/vamana/neighbors"


def make_vamana_graph(
        base_path: str,
        query_path: str,
        gt_path: str ,
        r: int,
        l: int,
        alpha: float,
        graph_outfile: str):
    """
    create a vamana graph in the [output_folder] with name
    """
    args = [
        vamana_binary_path,
        "-R",
        str(r),
        "-L",
        str(l),
        "-alpha",
        str(alpha), 
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        base_path,
        "-query_path",
        query_path,
        "-gt_path",
        gt_path,
        "-graph_outfile",
        graph_outfile,
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    alpha_int, alpha_decimal = str(alpha).split(".")
    result_file = f"vamana_{r}_{l}_{alpha_int}_{alpha_decimal}.txt"
    with open(result_file, "w") as f:
        f.write(res.stdout)
    print(res.stdout)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-base_path",
        help="path containing all the points",
        required=True)
    parser.add_argument(
        "-query_path", 
        help="path containing all the query points",
        required = True)
    parser.add_argument(
        "-gt_path",
        help="path containing all the groundtruth data for the query points",
        required = True)
    parser.add_argument(
        "-R",
        help="degree bound",
        required=True)
    parser.add_argument(
        "-L", 
        help="beamsearch length",
        required = True)
    parser.add_argument(
        "-alpha",
        help="alpha",
        required = True)
    parser.add_argument(
        "-graph_outfile",
        help="output filename for  graphs",
        required = True)
    # parser.add_argument(
        # "-result_outfile",
        # help="output filename for result",
        # required = True)

    args = parser.parse_args()
    print(args)
    make_vamana_graph(base_path=args.base_path,
                      query_path=args.query_path,
                      gt_path=args.gt_path,
                      r=args.R,
                      l=args.L,
                      alpha=args.alpha,
                      graph_outfile=args.graph_outfile)
                      
                      
                      
    
    













    
