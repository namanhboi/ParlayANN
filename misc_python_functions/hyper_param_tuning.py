import sys
import os
import argparse
from functools import partial
import subprocess
import optuna
from make_vamana_graph import make_vamana_graph

base_path = "/home/nd433/big-ann-benchmarks/data/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000"
query_path = "/home/nd433/big-ann-benchmarks/data/MSSPACEV1B/private_query_30k.bin"
gt_path = "/home/nd433/big-ann-benchmarks/data/MSSPACEV1B/msspacev-gt-1M"


# base_path = "/home/nd433/ParlayANN/data/sift/sift_learn.fbin"
# query_path = "/home/nd433/ParlayANN/data/sift/sift_query.fbin"
# gt_path = "/home/nd433/ParlayANN/data/sift/sift-100K"

vamana_binary_path = "/home/nd433/ParlayANN/algorithms/vamana/neighbors"


def objective(
        base_path,
        query_path,
        gt_path,
        gt_rank_path,
        metric,
        trial):
    R = trial.suggest_int("R", 10, 200)
    L = trial.suggest_int("L", 150, 210)

    alpha = trial.suggest_float("alpha", 1, 2, step=0.005)

    args = [
        "../data_tools/make_graph_run_queries",
        "-R",
        str(R),
        "-L",
        str(L),
        "-alpha",
        str(alpha),
        "-two_pass",
        "0",
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
        "-gt_rank_path",
        gt_rank_path,
        "-metric",
        metric
    ]
    
    results = subprocess.run(args, capture_output=True, text=True)
    num_hops = results.stdout.strip("\n").splitlines()[-1]
    (

        avg_total_hops,
        avg_phase_1_hops,
        avg_phase_2_hops,
        avg_total_time,
        avg_phase_1_time,
        avg_phase_2_time,
        recall,
        nn_beamsearch
    ) = [float(i) for i in num_hops.split(",")]
    metrics_dict = {
        "avg_total_hops": avg_total_hops,
        "avg_phase_1_hops": avg_phase_1_hops,
        "avg_phase_2_hops": avg_phase_2_hops,
        "avg_total_time": avg_total_time,
        "avg_phase_1_time": avg_phase_1_time,
        "avg_phase_2_time": avg_phase_2_time,
        "recall": recall,
        "nn_beamsearch" : nn_beamsearch
    }
    print(f"recall is {recall}, {metric} is {metrics_dict[metric]}")
    return metrics_dict[metric] if recall > 0.80 else 1



def write_best_trial_to_file(
        metric: str,
        best_params: dict,
        best_value: float,
        result_file: str):
    with open(result_file, 'a') as f:
        f.write(f"{metric},{best_params},{best_value}\n")


# if __name__ == "__main__":
    # write_best_trial_to_file("avg_phase_2_time", {"R": 99, "L": 74, "alpha": 1.0}, 5.946e-05, "result_tuned_avg_phase_2_time.txt")
    # make_vamana_graph(
    #             base_path=base_path,
    #             query_path=query_path,
    #             gt_path=gt_path,
    #             r=99,
    #             l=74,
    #             alpha=1.0,
    #             output_folder="../data/sift",
    #             extra_info=f"tuned_avg_phase_2_time")
    


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
        "-gt_rank_path",
        help="path containing all the groundtruth rank data for the query points",
        required = True)
    parser.add_argument(
        "-metric",
        help="metric to tune for",
        required=True,
        choices = [
            "avg_total_hops",
            "avg_phase_1_hops",
            "avg_phase_2_hops",
            "avg_total_time",
            "avg_phase_1_time",
            "avg_phase_2_time",
            "recall",
            "nn_beamsearch"],
        action = "append"
    )
    parser.add_argument(
        "-make_graph",
        help = "whether to make graphs with the best params after an objective finishes",
        action="store_true"
    )
    parser.add_argument(
        "-graph_folder",
        help="where to output the graph",
    )
    parser.add_argument(
        "-result_file",
        help="path to file to put the best params",
        required=True
    )
    parser.add_argument(
        "-num-trials",
        help="number of optuna trials to run per metric",
        type=int,
        default=50
    )

    args = parser.parse_args()
    if os.path.exists(args.result_file):
        should_delete = input("The specified result_file already exists, do you want to delete?[y/n]")
        if should_delete != "y" and should_delete != "n":
            print("please only enter y or n")
            sys.exit(1)
        if should_delete == "y":
            os.remove(args.result_file)

    for metric in args.metric:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            partial(
                objective,
                args.base_path,
                args.query_path,
                args.gt_path,
                args.gt_rank_path,
                metric),
            args.num_trials)
        write_best_trial_to_file(
            metric,
            study.best_params,
            study.best_value,
            args.result_file
        )
        if args.make_graph:
            make_vamana_graph(
                base_path=args.base_path,
                query_path=args.query_path,
                gt_path=gt_path,
                gt_rank_path=gt_rank_path,
                r=int(study.best_params["r"]),
                l=int(study.best_params["l"]),
                alpha=study.best_params["alpha"],
                output_folder=args.graph_folder,
                extra_info=f"tuned_{metric}")
                
