from argparse import ArgumentParser
from typing import Dict, Any

import torch
import os

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library
from seeslab_utils import expr_is_unique, torch_to_coords, expr_complexity, clean_folder
from plot_utils import plotData_plot, plot_avg, overlap_hists

# Plot data naming convention: plotData_xValues_yValues
# x: number of steps, y: number of unique expressions
def plotData_steps_nUniqExpr (step: int, walk_data):
    if "plotData_steps_nUniqExpr" not in walk_data:
        walk_data["plotData_steps_nUniqExpr"] = []
    n_uniq = walk_data["n_uniq_expr"]
    walk_data["plotData_steps_nUniqExpr"].append((step,n_uniq))


# x: number of steps without change, y: distance between changes
def plotData_stepsNoChng_chngDist (walk_data):
    data = [
        (step["steps_no_change"], step["change_distance"]) for step in walk_data["step_data"]
    ]
    walk_data["plotData_stepsNoChng_chngDist"] = data


#x: distance from origin, y: expression complexity
def plotData_norm_complexity(walk_data):
    data = []
    for step in walk_data["step_data"]:
        norm = step["norm"]
        complexity = expr_complexity(step["expression"])
        data.append((norm, complexity))
    walk_data["plotData_norm_complexity"] = data


# Histogram data naming convention: histData_xValues
# x: total number of visits for an expression
def histData_visits(walk_data):
    data = []
    expressions = walk_data["unique_expressions"]
    for expr in expressions.values():
        data.append(expr["visits"])
    walk_data["histData_visits"] = data


def save_step_data (step_data: Dict[str, Any], walk_data: Dict[str, Any]):
    # Saving the step data in the walk dictionary
    walk_data["step_data"].append(step_data)

    expressions = walk_data["unique_expressions"]
    expr = step_data["expression"]

    # Managing the uniquie expression if needed
    if (step_data["is_unique"] == True):
        walk_data["n_uniq_expr"] += 1
        expressions[expr] = {"expression": expr, "visits": 0}

    # Keeping track of the total number of visits of the expression so far
    expressions[expr]["visits"] += step_data["steps_no_change"] + 1


def walk_to_txt (path: str, walk_data: Dict[str, Any]):
    with open (path, 'w') as file:
        file.write(f"steps: {walk_data['steps']}\n")
        file.write(f"rand_range: {walk_data['rand_range']}\n\n")

        file.write(f"\nExpression || Steps without change || Distance to next change || Distance since start || Norm || Is Unique || Coordinates\n\n")
        for step_data in walk_data["step_data"]:
            file.write(f"{step_data['expression']} || {step_data['steps_no_change']} || {step_data['change_distance']} || {step_data['total_distance']} || {step_data['norm']} || {step_data['is_unique']} || {step_data['coords']}\n")


def random_walk(model, steps = 100) -> Dict[str, Any]:
    # Range for the generation of the random change in the vector
    a = 0.05
    b = -a

    # Initializing random walk dictionary
    walk_data = {
        "steps": steps,
        "rand_range": (min(a, b), max(a, b)),
        "step_data": [],
        "unique_expressions": {},
        "n_uniq_expr": 0
    }

    # Starting point
    lin = torch.randn(1,1,32)
    start_vector = lin.clone()

    # Initializing walk-tracking data
    current_expr = str(model.decode(lin)[0])
    current_vector = lin.clone()
    steps_no_change: int = 0
    total_distance: float = 0
    norm: float = torch.norm(lin[0,0]).item()
    coords = None

    # Implementing random walk
    for step in range(steps):
        # Adding a small random change between a and b
        delta = a * torch.randn(1,1,32)
        lin += delta
        new_expr = str(model.decode(lin)[0])

        # Checking for a change in the decoded expression
        if new_expr == current_expr:
            steps_no_change += 1
        if (current_expr != new_expr or step == steps - 1):
            # Computing distance between changes
            change_distance = torch.norm(lin[0,0] - current_vector[0,0]).item()

            # Saving data for the first step at which we obtained the current expression
            step_data = {
                "expression": current_expr,
                "steps_no_change": steps_no_change,
                "change_distance": change_distance,
                "total_distance": total_distance,
                "norm": norm,
                "is_unique": expr_is_unique(current_expr, walk_data),
                "coords": coords,
                "step_number": step
            }
            save_step_data(step_data, walk_data)

            # Updating walk-tracking variables
            current_expr = new_expr
            current_vector = lin.clone()
            steps_no_change = 0
            total_distance = torch.norm(lin[0,0] - start_vector[0,0]).item()
            norm = torch.norm(lin[0,0]).item()

            # Optional: Get the coordinates of the new expression
            # Replace None with torch_to_coords(lin) if needed
            coords = None

        # Optional: Generate plotting data (save specific data in (x,y)-tuples ready for plotting).
        # If unnecessary, comment the code
        plotData_steps_nUniqExpr(step,walk_data)
    plotData_stepsNoChng_chngDist(walk_data)
    plotData_norm_complexity(walk_data)
    histData_visits(walk_data)

    return walk_data

if __name__ == '__main__':
    parser = ArgumentParser(prog='Random Walk', description='Random walk for a specified amount of steps')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"], weights_only=False)

    steps = int(10000)
    results_path = f"../seeslab/random_walks/rw_ng_test"
    clean_folder(results_path)

    plots_steps_nuniq = []
    plots_stepsNoChng_chngDist = []
    plots_norm_complexity = []
    hists_to_overlap = []

    # Random Walk
    for i in range (1):
        walk_data = {}
        walk_data = random_walk(model, steps)

        # Converting walk data to txt
        txt_path = os.path.join(results_path, f"rw{i}.txt")
        walk_to_txt(txt_path, walk_data)

        # Plotting
        plot_path = os.path.join(results_path, f"plots{i}")
        plotData_plot(walk_data,plot_path)

        # Saving plots for computing average plot
        plots_steps_nuniq.append(walk_data["plotData_steps_nUniqExpr"])
        plots_stepsNoChng_chngDist.append(walk_data["plotData_stepsNoChng_chngDist"])
        plots_norm_complexity.append(walk_data["plotData_norm_complexity"])

        # Saving histograms to overlap
        hists_to_overlap.append(walk_data["histData_visits"])

    avgplots_path = os.path.join(results_path, "avg_plots")
    clean_folder(avgplots_path)

    plot_avg(plots_steps_nuniq, avgplots_path, "steps", "avg_nUniqExpr")
    plot_avg(plots_stepsNoChng_chngDist, avgplots_path, "stepsNoChng", "avg_chngDist")
    plot_avg(plots_norm_complexity, avgplots_path, "norm", "avg_compelxity")

    overlap_hists(hists_to_overlap, avgplots_path, "visits")
    overlap_hists(hists_to_overlap, avgplots_path, "visits", stacked=True)
    print("Random Walk Complete!")