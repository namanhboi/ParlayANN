import numpy as np
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import os


def graph_results(df: pd.DataFrame, filename):
    """
    create 5 graphs for each csv file
    """
    # plt.plot(df["point_index"], df["phase_1_freq"], 'o')
    fig, axes = plt.subplots(2, 3)

    q_hi_1 = df["query_num_hops_phase_1"].quantile(0.99)
    q_hi_2 = df["query_num_hops_phase_2"].quantile(0.99)

    df_filtered_1 = df[(df["query_num_hops_phase_1"] < q_hi_1)]
    df_filtered_2 = df[(df["query_num_hops_phase_2"] < q_hi_2)]
    print(df_filtered_1)

    axes[0, 0].hist(df_filtered_1["query_num_hops_phase_1"], alpha=0.5, label="p1")
    axes[0, 0].hist(df_filtered_2["query_num_hops_phase_2"], alpha=0.5, label="p2")

    axes[0, 0].set_xlim([0, 21])
    axes[0, 0].set_title("# hops in phases")
    print(max(df["query_num_hops_phase_2"]))
    axes[0, 0].legend()

    bins, bin_edges = pd.cut(df["query_time_phase_1"], bins=1000, retbins=True)
    axes[0, 1].hist(df["query_time_phase_1"], bins=bin_edges, alpha=0.5, label="p1")
    axes[0, 1].hist(df["query_time_phase_2"], bins=bin_edges, alpha=0.5, label="p2")
    axes[0, 1].set_xlim([-0.00001, 0.00015])
    axes[0, 1].set_title("Search time for phases")
    axes[0, 1].legend()

    axes[1, 0].scatter(
        df["point_index"][1:], df["phase_1_freq"][1:], label="phase 1 frequency", s=1
    )
    axes[1, 0].set_title("phase 1 frequency")
    axes[1, 1].scatter(
        df["point_index"],
        df["phase_2_freq"],
        label="phase 2 frequency",
        s=1,
        color="orange",
    )
    axes[1, 1].set_title("phase 2 frequency")

    axes[1, 2].scatter(df["point_index"][1:], df["node_freq"][1:], s=1, color="black")
    axes[1, 2].set_title("node freq")

    plt.tight_layout()
    try:
        os.remove(filename)
    except OSError:
         pass
    fig.savefig(filename, dpi=300)


def overlayed_phases_hops_graph(df_list: list[pd.DataFrame], filename: str):
    fig, axes = plt.subplots(ncols = 1, nrows = 2, figsize = (7, 10))
    _, fake_axes = plt.subplots(ncols = 1, nrows = 2, figsize = (7, 10))
    q_hi_1_list = [df["query_num_hops_phase_1"].quantile(0.99) for df in df_list]
    q_hi_2_list = [df["query_num_hops_phase_2"].quantile(0.99) for df in df_list]

    df_filtered_1_list = [
        df[(df["query_num_hops_phase_1"] < q_hi_1_list[i])]["query_num_hops_phase_1"]
        for i, df in enumerate(df_list)
    ]
    df_filtered_2_list = [
        df[(df["query_num_hops_phase_2"] < q_hi_2_list[i])]["query_num_hops_phase_2"]
        for i, df in enumerate(df_list)
    ]
    print(df_filtered_1_list)

    # axes[0, 0].hist(df_filtered_1["query_num_hops_phase_1"], alpha=0.5, label='p1')
    # axes[0, 0].hist(df_filtered_2["query_num_hops_phase_2"], alpha=0.5, label='p2')

    # axes[0, 0].set_xlim([0, 21])
    # axes[0, 0].set_title("# hops in phases")
    # print(max(df["query_num_hops_phase_2"]))
    # axes[0, 0].legend()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[0].set_xlim([0, 18])
    axes[1].set_xlim([0, 18])
    og_y_1 = 15000
    y_diff_1 = 6000

    og_y_2 = 20000
    y_diff_2 = 8000


    for i in range(len(df_list)):
        n_1, x_1, _ = axes[0].hist(
            df_filtered_1_list[i], alpha=0.5, label=df_list[i].name + " phase 1", color = colors[i], range = (0, 18), bins = 18
        )
        bin_centers_1 = 0.5 * (x_1[1:] + x_1[:-1])
        axes[0].plot(bin_centers_1, n_1,  color=colors[i], label = None)
        axes[0].text(13, og_y_1 - y_diff_1 * i, df_list[i].name + "\n" + df_filtered_1_list[i].describe().loc[['mean', 'std']].to_string(), color = colors[i])
        n_2, x_2, _ = axes[1].hist(
            df_filtered_2_list[i], alpha=0.5, label=df_list[i].name + " phase 2", color = colors[i], range = (0, 18), bins = 18
        )
        bin_centers_2 = 0.5 * (x_2[1:] + x_2[:-1])
        print(bin_centers_2)
        axes[1].plot(bin_centers_2, n_2, color=colors[i], label = None)
        axes[1].text(1, og_y_2 - y_diff_2 * i, df_list[i].name + "\n" +  df_filtered_2_list[i].describe().loc[['mean', 'std']].to_string(), color = colors[i])
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel("number of hops")
    axes[1].set_xlabel("number of hops")
    axes[0].set_title("number of hops in phase 1")
    axes[1].set_title("number of hops in phase 2")

    plt.tight_layout()

    try:
        os.remove(filename)
    except OSError:
        pass
    fig.savefig(filename, dpi=300)


def scatter_plot_queries_against_time(df_list, filename:str):
    fig, axes = plt.subplots(1)
    axes.scatter(df_list[0]["point_index"], np.log(df_list[0]["query_time_phase_1"]), alpha = 0.1)
    axes.scatter(df_list[0]["point_index"], np.log(df_list[0]["query_time_phase_2"]), alpha = 0.1)
    fig.savefig(filename, dpi=300)
    


def overlayed_phases_time_graph(
        df_list: list[pd.DataFrame],
        filename: str,
        xlim: list[float]):
    fig, axes = plt.subplots(ncols = 1, nrows = 4, figsize = (7, 15))
    _, fake_axes = plt.subplots(ncols = 1, nrows = 4, figsize = (7, 15))
    bins_1, bin_edges_1 = pd.cut(
        df_list[0]["query_time_phase_1"], bins=1000, retbins=True
    )
    bins_2, bin_edges_2 = pd.cut(
        df_list[0]["query_time_phase_2"], bins=1000, retbins=True
    )
    bins_total, bin_edges_total = pd.cut(
        df_list[0]["query_time_total"], bins=1000, retbins=True
    )
    bins_nn_beamsearch, bin_edges_nn_beamsearch = pd.cut(
        df_list[0]["nn_beamsearch_time"], bins=1000, retbins=True
    )
    colors = ["r", "g", "b", "yellow"]
    y_space_phase_1 = 500
    og_y_space_phase_1 =3000
    y_space_phase_2 = 2000
    og_y_space_phase_2 = 8000
    y_space_phase_total = 2000
    og_y_space_phase_total = 9000
    y_space_nn_beamsearch = 3000
    og_y_space_nn_beamsearch = 10000
    
    for i in range(len(df_list)):
        n_1, x_1, _ = fake_axes[0].hist(
            df_list[i]["query_time_phase_1"],
            bins=bin_edges_1,
            alpha=0.5,
            label=df_list[i].name,
            color=colors[i],
        )
        bin_centers_1 = 0.5 * (x_1[1:] + x_1[:-1])
        axes[0].plot(bin_centers_1, n_1, label=df_list[i].name, color=colors[i])
        # axes[0].text(0.00010, og_y_space_phase_1 - y_space_phase_1 * i, df_list[i].name + "\n" + df_list[i]["query_time_phase_1"].describe().loc[['mean', 'std']].to_string(), color = colors[i])
        n_2, x_2, _ = fake_axes[1].hist(
            df_list[i]["query_time_phase_2"],
            bins=bin_edges_2,
            alpha=0.5,
            label=df_list[i].name,
            color=colors[i],
        )
        bin_centers_2 = 0.5 * (x_2[1:] + x_2[:-1])
        axes[1].plot(bin_centers_2, n_2, label=df_list[i].name, color=colors[i])
        # axes[1].text(0.0001, og_y_space_phase_2 - y_space_phase_2 * i, df_list[i].name + '\n'  + df_list[i]["query_time_phase_2"].describe().loc[['mean', 'std']].to_string(), color = colors[i])

        n_total, x_total, _ = fake_axes[2].hist(
            df_list[i]["query_time_total"],
            bins=bin_edges_total,
            alpha=0.5,
            label=df_list[i].name,
            color=colors[i],
        )
        bin_centers_total = 0.5 * (x_total[1:] + x_total[:-1])
        axes[2].plot(bin_centers_total, n_total, label=df_list[i].name, color=colors[i])
        # axes[2].text(0.0001, og_y_space_phase_total - y_space_phase_total * i, df_list[i].name + '\n'  + df_list[i]["query_time_total"].describe().loc[['mean', 'std']].to_string(), color = colors[i])

        n_nn_beamsearch, x_nn_beamsearch, _ = fake_axes[3].hist(
            df_list[i]["nn_beamsearch_time"],
            bins=bin_edges_total,
            alpha=0.5,
            label=df_list[i].name,
            color=colors[i],
        )
        bin_centers_nn_beamsearch = 0.5 * (x_nn_beamsearch[1:] + x_nn_beamsearch[:-1])
        axes[3].plot(bin_centers_nn_beamsearch, n_nn_beamsearch, label=df_list[i].name, color=colors[i])
        # axes[3].text(0.0001, og_y_space_nn_beamsearch - y_space_nn_beamsearch * i, df_list[i].name + '\n'  + df_list[i]["nn_beamsearch_time"].describe().loc[['mean', 'std']].to_string(), color = colors[i])

    axes[0].set_xlim(xlim)
    axes[1].set_xlim(xlim)
    axes[2].set_xlim(xlim)
    axes[3].set_xlim(xlim)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[0].set_xlabel("search time spent in phase 1")
    axes[1].set_xlabel("search time spent in phase 2")
    axes[2].set_xlabel("search time spent in total")
    axes[3].set_xlabel("search time spent in nn beamsearch")
    try:
        os.remove(filename)
    except OSError:
        pass
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input-path", help="path to csv file", nargs="+")
    parser.add_argument("-op", "--output-path", help="path to the png file", required = True)
    parser.add_argument("-t", "--type", help = "type of graph you want to make: all_data_one_graph means that you plot hops, time, etc data for just 1 graph; hops means that you can compare the number of hops of each graph for every query, likewise for time", choices = ["all_data_one_graph", "hops", "time", "scatter"], required=True)
    
    args = parser.parse_args()
    if args.type == "all_data_one_graph":
        if len(args.input_path) > 1:
            print("only one graph allowed")
            sys.exit(1)
        df = pd.read_csv(args.input_path)
        graph_results(df, args.output_path)
    elif args.type == "hops":
        df_list = [pd.read_csv(csv_file) for csv_file in args.input_path]
        for i in range(len(args.input_path)):
            df_list[i].name = Path(args.input_path[i]).stem
        overlayed_phases_hops_graph(df_list, args.output_path)
    elif args.type == "time":
        df_list = [pd.read_csv(csv_file) for csv_file in args.input_path]
        for i in range(len(args.input_path)):
            df_list[i].name = Path(args.input_path[i]).stem
        overlayed_phases_time_graph(
            df_list,
            args.output_path,
            [0.0, 0.009]
        )

    elif args.type == "scatter":
        df_list = [pd.read_csv(csv_file) for csv_file in args.input_path]
        for i in range(len(args.input_path)):
            df_list[i].name = Path(args.input_path[i]).stem
        scatter_plot_queries_against_time(df_list, "scatter.png")
        

    # # csv_files = [
    # #     "/home/nam/vector_index_rs/ParlayANN/data_tools/phase_detection_result_hcnng_3_10.csv",
    # #     "/home/nam/vector_index_rs/ParlayANN/data_tools/phase_detection_result_pynndescent_30.csv",
    # #     "/home/nam/vector_index_rs/ParlayANN/data_tools/phase_detection_result_vamana_32_64.csv",
    # # ]
    # csv_files = [
    #     "/home/nam/vector_index_rs/ParlayANN/data_tools/phase_detection_result_vamana_92_25.csv",
    #     "/home/nam/vector_index_rs/ParlayANN/data_tools/phase_detection_result_vamana_100_133.csv",
    #     ]
    # df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    # df_list[0].name = "vamana_92_25"
    # df_list[1].name = "vamana_100_133"
    # graph_filename = "overlayed_vamana_tuned_hops.png"
    # # overlayed_phases_time_graph(df_list, graph_filename)
    # overlayed_phases_hops_graph(df_list, graph_filename)
