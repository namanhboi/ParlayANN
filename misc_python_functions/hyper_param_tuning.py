from functools import partial
import subprocess
import optuna

base_path = "../data/sift/sift_learn.fbin"
query_path = "../data/sift/sift_query.fbin"
gt_path = "../data/sift/sift-100K"


def objective(metric, trial):
    R = trial.suggest_int("R", 1, 100)
    L = trial.suggest_int("L", 25, 200)
    alpha = trial.suggest_float("alpha", 0.75, 2, step=0.005)

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
    ) = [float(i) for i in num_hops.split(",")]
    metrics_dict = {
        "avg_total_hops": avg_total_hops,
        "avg_phase_1_hops": avg_phase_1_hops,
        "avg_phase_2_hops": avg_phase_2_hops,
        "avg_total_time": avg_total_time,
        "avg_phase_1_time": avg_phase_1_time,
        "avg_phase_2_time": avg_phase_2_time,
        "recall": recall,
    }
    return metrics_dict["metric"] if recall > 0.9 else 100000


if __name__ == "__main__":
    # R = 32
    # L = 64
    # alpha = 1.2
    # args = [
    #     "../data_tools/make_graph_run_queries",
    #     "-R",
    #     str(R),
    #     "-L",
    #     str(L),
    #     "-alpha",
    #     str(alpha),
    #     "-two_pass",
    #     "0",
    #     "-data_type",
    #     "float",
    #     "-dist_func",
    #     "Euclidian",
    #     "-base_path",
    #     base_path,
    #     "-query_path",
    #     query_path,
    #     "-gt_path",
    #     gt_path
    # ]
    # print(args)
    # results = subprocess.run(args, capture_output=True, text=True)
    # print(results.stdout)
    
    study_total_time = optuna.create_study(direction="minimize")
    study_total_time.optimize(partial(objective, "avg_total_time"), n_trials=50)
    print("total hops best params: ", study_total_time.best_params)

    study_phase_1_time = optuna.create_study(direction="minimize")
    study_phase_1_time.optimize(partial(objective, "avg_phase_1_time"), n_trials=50)
    print("phase 1 hops best params: ", study_phase_1_time.best_params)

    study_phase_2_time = optuna.create_study(direction="minimize")
    study_phase_2_time.optimize(partial(objective, "avg_phase_2_time"), n_trials=50)
    print("phase 2 hops best params: ", study_phase_2_time.best_params)
