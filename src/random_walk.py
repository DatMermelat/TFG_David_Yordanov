from argparse import ArgumentParser

import numpy as np
import torch
import os

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library
from seeslab_utils import torch_to_coords, expr_complexity, clean_folder
from plot_utils import generate_plots, plot_avg, overlap_hists

class RandomWalk:
    def __init__(self, model, symbols: dict, steps: int = 1000, step_size: float = 0.05):
        self.model = model
        self.symbols = symbols
        self.steps = steps
        self.step_size = step_size
        self.walk_data = []
        self.expressions = {}
        self.n_unique = 0


    def is_unique(self, expr: str) -> bool:
        return expr not in self.expressions

    def plots(self, path: str = None):
        self.plots = []
        steps = [step["step"] for step in self.walk_data]
        complexities = [step["complexity"] for step in self.walk_data]
        norms = [step["norm"] for step in self.walk_data]
        chng_dists = [step["change_distance"] for step in self.walk_data]
        steps_no_change = [step["steps_no_change"] for step in self.walk_data]
        first_visits = [self.expressions[expr]["first_visit"] for expr in self.expressions]
        total_visits = [self.expressions[expr]["total_visits"] for expr in self.expressions]

        self.plots.append({
            "type": "scatter",
            "x": steps,
            "y": complexities,
            "title": "Complexity vs Steps",
            "xlabel": "Steps",
            "ylabel": "Decoded Expression Complexity"
        })

        self.plots.append({
            "type": "scatter",
            "x": steps,
            "y": norms,
            "title": "Norm vs Steps",
            "xlabel": "Steps",
            "ylabel": "Norm"
        })

        self.plots.append({
            "type": "scatter",
            "x": steps_no_change,
            "y": chng_dists,
            "title": "Change Distance vs Steps Without Change",
            "xlabel": "Steps Without Change",
            "ylabel": "Change Distance"
        })

        self.plots.append({
            "type": "scatter",
            "x": steps,
            "y": chng_dists,
            "title": "Change Distance vs Steps",
            "xlabel": "Steps",
            "ylabel": "Change Distance"
        })

        self.plots.append({
            "type": "line",
            "x": first_visits,
            "y": list(range(1,self.n_unique+1)),
            "title": "Unique expressions vs Steps",
            "xlabel": "Steps",
            "ylabel": "Number of Unique Expressions"
        })

        self.plots.append({
            "type": "histogram",
            "data": total_visits,
            "bins": "auto",
            "title": "Visits distribution",
            "xlabel": "log10(Visits)",
            "ylabel": "Relative Frequency",
        })

        generate_plots(self.plots, path=path)

    def to_txt (self, path: str):
        with open (path, 'w') as file:
            file.write(f"Expression || Steps without change || Distance to next change || Norm\n")
            for step_data in self.walk_data:
                file.write(f"{step_data['expr']} || {step_data['steps_no_change']} || {step_data['change_distance']} || {step_data['norm']}\n")

    def run(self, from_origin: bool = True):
        # Starting point
        lin = torch.zeros(1, 1, 32) if from_origin else self.step_size * torch.randn(1, 1, 32)

        # Initializing walk-tracking data
        current_expr = model.decode(lin)[0]
        current_vector = lin.clone()
        steps_no_change: int = 0
        norm: float = torch.norm(lin).item()

        # Implementing random walk
        for step in range(self.steps):
            # Adding a small random increment
            delta = self.step_size * torch.randn(1,1,32)
            lin += delta
            new_expr = model.decode(lin)[0]

            # Checking for a change in the decoded expression
            if str(new_expr) == str(current_expr):
                steps_no_change += 1
            if str(current_expr) != str(new_expr) or step == self.steps - 1:
                # Computing distance between changes
                change_distance = torch.norm(lin - current_vector).item()

                # Saving data for the first step at which we obtained the current expression
                step_data = {
                    "step": step,
                    "expr": str(current_expr),
                    "complexity": expr_complexity(current_expr, self.symbols),
                    "steps_no_change": steps_no_change,
                    "change_distance": change_distance,
                    "norm": norm,
                }
                self.walk_data.append(step_data)

                if self.is_unique(str(current_expr)):
                    self.n_unique += 1
                    self.expressions[str(current_expr)] = {
                        "first_visit": step,
                        "total_visits": 1
                    }
                else:
                    self.expressions[str(current_expr)]["total_visits"] += 1

                # Updating walk-tracking variables
                current_expr = new_expr
                current_vector = lin.clone()
                steps_no_change = 0
                norm = torch.norm(lin).item()



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

    steps = 10000
    runs = 10
    results_path = f"../seeslab/random_walks/rw_test"
    clean_folder(results_path)

    for run in range(runs):
        rw = RandomWalk(model, so, steps=steps, step_size=0.05)
        rw.run(from_origin=True)
        rw.plots(os.path.join(results_path, f"plots{run}"))
        rw.to_txt(os.path.join(results_path, f"rw{run}.txt"))

    print("Random Walk Complete!")