import numpy as np
import random
import math
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.cm as cm
cmap = cm.get_cmap("Dark2", 100)
colors =  [cmap(i) for i in range(100)]


def _get_phase_separation_index(frontier: list[list[int]]):
    phase_separation_index = 0
    for i in range(len(frontier)):
        if any([idx in frontier[-1] for idx in frontier[i]]):
            return i
    return phase_separation_index

def _plot_query_info(ax,
                     log1p_distance_to_query_rank: list[float],
                     phase_separation_index: int,
                     alpha, color):
    # ax.set_xlim(0, 15)
    ax.plot(np.arange(len(log1p_distance_to_query_rank)), log1p_distance_to_query_rank, "x-", alpha=alpha, color=color)
    ax.axvline(phase_separation_index, linestyle='-', alpha=alpha, color=color)


    
def _summary_stat_phase_separation(df):
    phase_1_len_list = []
    phase_2_len_list = []
    total_len = []
    for query_index in range((df["query_index"].nunique())):
        current_frontier = df[df["query_index"] == query_index]["frontier"].to_list()
        phase_1_len = _get_phase_separation_index(current_frontier)
        phase_2_len = df[df["query_index"]==query_index].count() - phase_1_len
        phase_1_len_list.append(phase_1_len)
        phase_2_len_list.append(phase_2_len)
        total_len.append(len(df[df["query_index"] == query_index]["frontier"].to_list()))
    return np.mean(phase_1_len_list), np.median(phase_1_len_list), np.std(phase_1_len_list), np.mean(phase_2_len_list), np.median(phase_2_len_list), np.std(phase_2_len_list), np.mean(total_len), np.median(total_len), np.std(total_len)
        

def make_two_phase_graph(output_path: str,
                         df: pd.DataFrame,
                         num_lines: str,
                         title: str):
    random_query_indices = random.sample(range(0, df["query_index"].nunique()), num_lines)
    cols = math.ceil(math.sqrt(num_lines))
    rows = math.ceil(num_lines / cols)
    fig, ax_list = plt.subplots(cols, rows, figsize=(cols * 2, rows * 2))
    ax_list = ax_list.flatten()
    alpha = 1
    for i, query_index in enumerate(random_query_indices):
        query_info_df = df[df["query_index"] == query_index]
        phase_separation_index = _get_phase_separation_index(query_info_df["frontier"].to_list())
        _plot_query_info(ax_list[i],
                         query_info_df["log1p_distance_to_query_rank"].to_list(),
                         phase_separation_index,
                         alpha, colors[i])
    

    mean_phase_1_hops, median_phase_1_hops, std_phase_1_hops, mean_phase_2_hops, median_phase_2_hops, std_phase_2_hops, mean_total_hops, median_total_hops, std_total_hops =   _summary_stat_phase_separation(df)
    
    fig.text(0.05, 0.92, f'mean_phase_1:{mean_phase_1_hops}\nmedian_phase_1_hops:{median_phase_1_hops}\nstd_phase_1_hops:{std_phase_1_hops}', ha='left', va='center', fontsize=25)
    fig.text(0.5, 0.92, f'mean_phase_1:{mean_phase_2_hops}\nmedian_phase_2_hops:{median_phase_2_hops}\nstd_phase_2_hops:{std_phase_2_hops}', ha='left', va='center', fontsize=25)
    fig.text(0.5, 0.02, f'mean_total:{mean_total_hops}\nmedian_total_hops:{median_total_hops}\nstd_total_hops:{std_total_hops}', ha='left', va='center', fontsize=25)


    # mean_phase_2_hops:{mean_phase_2_hops};median_phase_2_hops:{median_phase_2_hops};std_phase_2_hops:{std_phase_2_hops}'
    fig.suptitle(title, fontsize=30)
    fig.savefig(output_path, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--num-lines", type=int, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df["frontier"] = df["frontier"].apply(lambda x: literal_eval(x))
    df["log1p_distance_to_query_rank"] = df["distance_to_query_rank"].apply(lambda x: math.log(x + 1))


    make_two_phase_graph(args.output_file, df, args.num_lines, args.title)
