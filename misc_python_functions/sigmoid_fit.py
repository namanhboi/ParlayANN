import argparse
from scipy.optimize import curve_fit
import numpy as np
import math


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def parse_input_file(path: str):
    """
    parses file generated by vbase_graph.cpp and returns data for graph
    """
    with open(path, "r") as f:
        # command = f.readline()
        # query_index = int(f.readline())
        lines = [line.rstrip().split(",") for line in f]

    x_axis, y_axis = list(zip(*lines))
    x_axis_values = [int(i) for i in x_axis]
    y_axis_values = [math.log(int(i) + 1) for i in y_axis]
    # y_axis_values = [int(i) for i in y_axis]
    return x_axis_values, y_axis_values


def fit_sigmoid(x_data, y_data):
    p0 = [
        max(y_data),
        np.median(x_data),
        1,
        min(y_data),
    ]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method="dogbox")
    return popt, pcov


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-p", "--path", help="path to the input file generated by vbase_graph.cpp"
#     )
#     args = parser.parse_args()
#     path = args.path
#     x_axis, y_axis = parse_input_file(path)
#     fit_sigmoid(x_axis, y_axis)
