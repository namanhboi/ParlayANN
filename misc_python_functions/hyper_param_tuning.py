import subprocess
import optuna


def objective_phase_2_hops(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
    ]
    results = subprocess.run(args, capture_output=True, text=True)
    num_hops = results.stdout.strip("\n").splitlines()[-1]
    avg_total_hops, avg_phase_1_hops, avg_phase_2_hops = [
        float(i) for i in num_hops.split(",")
    ]
    return avg_phase_2_hops


def objective_phase_1_hops(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
    ]
    results = subprocess.run(args, capture_output=True, text=True)
    num_hops = results.stdout.strip("\n").splitlines()[-1]
    avg_total_hops, avg_phase_1_hops, avg_phase_2_hops = [
        float(i) for i in num_hops.split(",")
    ]
    return avg_phase_1_hops


def objective_total_hops(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
    ]
    results = subprocess.run(args, capture_output=True, text=True)
    num_hops = results.stdout.strip("\n").splitlines()[-1]
    avg_total_hops, avg_phase_1_hops, avg_phase_2_hops = [
        float(i) for i in num_hops.split(",")
    ]
    return avg_total_hops


def objective_time_total(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
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
    ) = [float(i) for i in num_hops.split(",")]
    return avg_total_time

def objective_time_phase_1(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
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
    ) = [float(i) for i in num_hops.split(",")]
    return avg_phase_1_time


def objective_time_phase_2(trial):
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
        "-data_type",
        "float",
        "-dist_func",
        "Euclidian",
        "-base_path",
        "../data/sift/sift_learn.fbin",
        "-num_query 10000",
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
    ) = [float(i) for i in num_hops.split(",")]
    return avg_phase_2_time


if __name__ == "__main__":
    study_total_hops = optuna.create_study(direction="minimize")
    study_total_hops.optimize(objective_time_total, n_trials=50)
    print("total hops best params: ", study_total_hops.best_params)

    study_phase_1_hops = optuna.create_study(direction="minimize")
    study_phase_1_hops.optimize(objective_time_phase_1, n_trials=50)
    print("phase 1 hops best params: ", study_phase_1_hops.best_params)

    study_phase_2_hops = optuna.create_study(direction="minimize")
    study_phase_2_hops.optimize(objective_time_phase_2, n_trials=50)
    print("phase 2 hops best params: ", study_phase_2_hops.best_params)
