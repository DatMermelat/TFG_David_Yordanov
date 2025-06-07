from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from seeslab_utils import clean_folder


def get_plot_fig (x_values: list, y_values: list, x_label: str = "x", y_label: str = "y", title: str = None, scatter=False):
    if scatter:
        plt.scatter(x_values, y_values, s=0.8)
    else:
        plt.plot(x_values, y_values)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + "-" + y_label)
    fig = plt.gcf()
    plt.show()
    plt.clf()
    plt.close()
    return fig


def get_hist_fig (data: list, x_label: str = "x", bins = None):
    plt.hist(data, bins = bins, color='blue', alpha=0.7, density = False)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(x_label + " distribution")
    fig = plt.gcf()
    plt.show()
    plt.clf()
    plt.close()
    return fig


# Method for overlapping a set of histograms
def overlap_hists(hists: list, path: str, x_label: str = "all_values", stacked: bool = False):
    type = "stacked" if stacked else "overlap"
    hist_path=os.path.join(path, type + "_" + x_label + ".png")

    bins = np.logspace(0, 3, base=10, num=18)
    plt.hist(hists, bins=bins, alpha = 0.7, density = False, stacked=stacked)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(hist_path, format="png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


# Method for overlappin a set of line-plots
def overlap_plots(plots: list, path: str = None, x_label: str = "x", y_label: str = "y"):
    palette = sns.color_palette("husl", n_colors=len(plots))

    plt.figure(figsize=(10, 6))
    for i, plot in enumerate(plots):
        x_values, y_values = zip(*plot)
        sns.lineplot(x=x_values, y=y_values, label=f"Run {i}", color=palette[i])

    plt.xscale('log')
    plt.yscale('log')

    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Saving the plot
    if path:
        plot_path = os.path.join(path, f"{x_label}_{y_label}.png")
        plt.savefig(plot_path, format="png", dpi=150, bbox_inches='tight')
        print(f"Plot saved at: {plot_path}")
    plt.show()
    plt.close()


def generate_plots (plots: list, path: str):
    """
    Generates plot figures based on the provided plot specifications.
    """

    for plot in plots:
        plt.figure(figsize=(8, 5))

        # Plot type
        if plot["type"] == "scatter":
            sns.regplot(x=plot["x"], y=plot["y"], scatter=True, line_kws={"color": "red"}, scatter_kws={"s": 10})
            plt.xlim(left=0)
        elif plot["type"] == "line":
            sns.lineplot(x=plot["x"], y=plot["y"])
        elif plot["type"] == "histogram":
            discrete = False if "bins" in plot else True
            sns.histplot(plot["data"], bins=plot.get("bins", "auto"), discrete=discrete, stat="probability")
        else:
            raise ValueError(f"Unknown plot type: {plot['type']}")

        # Titles and labels
        plt.title(plot.get("title", ""))
        plt.xlabel(plot.get("xlabel", ""))
        plt.ylabel(plot.get("ylabel", ""))

        # Optional axis scaling
        if plot.get("xscale"):
            plt.xscale(plot["xscale"])
        if plot.get("yscale"):
            plt.yscale(plot["yscale"])

        # Save plot with a clear filename
        filename = f"{plot.get('title', 'plot').replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(path, filename))
        plt.close()

def plot_avg(plots: list, path: str = None, x_label: str = "x", y_label: str = "avg_y", scatter = False, ci=95):
    # Flatten the data into a DataFrame
    data = []
    for plot in plots:
        for x, y in plot:
            data.append((x, y))
    df = pd.DataFrame(data, columns=[x_label, y_label])

    # Use seaborn to plot the average with confidence intervals
    plt.figure(figsize=(10, 6))
    if scatter:
        sns.scatterplot(data=df, x=x_label, y=y_label, s=20)
    else:
        sns.lineplot(data=df, x=x_label, y=y_label)

    # plt.xscale('log')
    plt.yscale('log')

    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Saving the plot
    if path:
        plot_path = os.path.join(path, f"{x_label}_{y_label}.png")
        plt.savefig(plot_path, format="png", dpi=150, bbox_inches='tight')
        print(f"Plot saved at: {plot_path}")

    plt.show()
    plt.close()