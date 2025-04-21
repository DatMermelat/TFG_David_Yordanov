from argparse import ArgumentParser

import numpy as np
import torch
import os
import copy

from model import HVAE
from hvae_utils import load_config_file, tokens_to_tree
from symbol_library import generate_symbol_library
from seeslab_utils import distance, clean_folder
from plot_utils import plotData_plot, overlap_plots, plot_avg
from evaluation import RustEval
from tree import Node

class RegressionData:

    def __init__(self):
        self.data = {}
        self.best = None
        self.plots = {}
        self.step = 0


    def save_expr_data(self, iteration: int, mse: float, expr: Node, coords: torch.Tensor):
        self.data[self.step]={"iteration": iteration, "error": mse, "expr": expr, "coords": coords.clone()}
        self.step += 1


    def save_target(self, expr: Node, coords: torch.Tensor):
        self.data["target"]={"expr": expr, "coords": coords}


    def steps(self):
        return self.step


    def set_best(self, best_mse: float, best_expr: Node, coords: torch.Tensor):
        self.best = {"error": best_mse, "expr": best_expr, "coords": coords.clone()}


    def plotData_steps_distToTrgt(self):
        """
        Generates a list of (step, distance_to_target) tuples
        """
        points = [
            (step, distance(self.data[step]["coords"], self.data["target"]["coords"])) for step in self.data if step != "target"
        ]
        self.plots["plotData_steps_distToTrgt"] = points


    def plotData_distToTrgt_error(self):
        """
        Generates a list of (mean_squared_error, distance_to_target) tuples
        """
        points = [
            (distance(self.data[step]["coords"], self.data["target"]["coords"]), self.data[step]["error"]) for step in self.data if step != "target"
        ]
        self.plots["plotData_distToTrgt_error"] = points


    def plotData_steps_error(self):
        """
        Generates a list of (step, mean_squared_error) tuples
        """
        points = [(self.data[step]["iteration"], self.data[step]["error"]) for step in self.data if step != "target"]
        self.plots["plotData_steps_error"] = points


    def plot_all(self, path: str):
        self.plotData_steps_error()
        # self.plotData_distToTrgt_error()
        # self.plotData_steps_distToTrgt()
        plotData_plot(self.plots, path)


    def to_txt(self, path, show_coords=False):
        with open(path,'w') as file:
            for step in self.data:
                if step != "target":
                    text = f"{step} || {self.data[step]['iteration']} || {self.data[step]['error']} || {str(self.data[step]['expr'])}"
                    if show_coords:
                        coords = self.data[step]['coords'].flatten().tolist()
                        file.write(text + f" || {coords}\n")
                    else:
                        file.write(text + "\n")
            # Writing best solution
            if self.best is not None:
                file.write(f"best: {self.best['error']} || {self.best['expr']}")


def get_expr_mse(evaluator: RustEval, expr: Node):
    RPN_expr = expr.to_list("postfix")
    rmse = np.array(evaluator.get_error(RPN_expr))
    mse = rmse**2
    return mse.tolist()


def generate_data(model, expr=None, indep_values: list=None, std_dev: float=1.0) -> np.array:
    '''
    INPUT:
    expr (tree.Node): The target expression. If None, a random expression will be sampled from the latent space.
    indep_values (list): List of independent variable values.
    std_dev (float): Standard deviation for the sampling of a random expression. Default distribution is N(0,1).

    OUTPUT:
    data (np.array): Processed array with independent and target values. Ready for the use of RustEval.
    expr (tree.Node): The expression from wich the target values were derived.
    coords (torch.Tensor): A 1x1xN (N=latent space dimensions) tensor with the latent space coordinates of the target.
    '''
    if indep_values is None:
        indep_values = np.linspace(1,10, num=1000)

    target = None
    coords = None
    if expr is None:
        while target is None:
            coords = std_dev*torch.randn(1,1,32)
            expr = model.decode(coords)[0]
            evaluator = RustEval(np.array([indep_values]), no_target=True)
            target = evaluator.evaluate(expr.to_list("postfix"))
    else:
        evaluator = RustEval(np.array([indep_values]), no_target=True)
        target = evaluator.evaluate(expr.to_list("postfix"))
        if target is None:
            raise ValueError("Selected target is not defined for the entire domain of independent values.")

    data = np.array([indep_values,target[0]])
    print(f"Target expression: {expr}\n")
    return expr, data, coords


def set_starting_point(model, evaluator, from_origin=False):
    start_vector = torch.zeros(1,1,32) if from_origin else torch.randn(1,1,32)
    start_expr = model.decode(start_vector)[0]
    start_mse = min(get_expr_mse(evaluator, start_expr))
    return start_vector, start_expr, start_mse


def mse_regression(model, data: np.array, tolerance = 1e-15, max_iter: int=10000, from_origin=False) -> RegressionData:
    evaluator = RustEval(data, default_value = float('inf'), verbose = False)
    results = RegressionData()

    # Setting starting point
    best_vector, best_expr, best_mse = set_starting_point(model, evaluator, from_origin)
    current_expr = best_expr
    eval_vector = best_vector.clone()

    # Safety Check (No division by 0, log of a negative number, etc.)
    if best_mse < float('inf'):
        results.save_expr_data(iteration=0, mse=best_mse, expr=best_expr, coords=best_vector)
        print(f"start: {best_expr}   mse: {best_mse}\n")
    else:
        if from_origin:
            raise ValueError("Invalid starting point. Restart the run with different independent values or set 'from_origin' to False.")
        else:
            raise ValueError("Invalid starting point. Restart run.")

    # Regression
    i = 1
    while best_mse > tolerance and i <= max_iter:
        # Taking random steps until there is a change in the decoded expression
        while str(current_expr) == str(best_expr):
            delta = 0.3 * torch.randn(1,1,32)
            eval_vector += delta
            current_expr = model.decode(eval_vector)[0]

        # Evaluating expression and calculating mse
        current_mse = min(get_expr_mse(evaluator, current_expr))

        # Checking if the change is acceptable
        if current_mse <= best_mse: # Accept and update best result
            results.save_expr_data(iteration=i, mse=current_mse, expr=current_expr, coords=eval_vector)
            best_mse = current_mse
            best_expr = current_expr
            best_vector = eval_vector.clone()
            # print(f"accepted: {current_expr}   mse: {current_mse}\n")
        else: # Reject and go back to the best expression
            eval_vector = best_vector.clone()
            current_expr = best_expr

        # Print progress
        if i % 1000 == 0:
            print(f"Progress: {i/max_iter*100: .1f}%", end='\r')
        i += 1

    if best_mse <= tolerance:
        print(f"MSE converged to tolerance {tolerance} in {results.steps()} steps over {i-1} iterations.\n")
    else:
        print(f"Stopped after reaching max iterations ({max_iter}).\n")
    return results


if __name__ == '__main__':
    parser = ArgumentParser(prog='MSE Regression', description='Symbolic regression by minimizing the MSE')
    parser.add_argument("-config", default="../configs/test_config.json")
    parser.add_argument("-runs", default=1)
    parser.add_argument("-targets", default=1)

    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"], weights_only=False)

    ng_expressions = [
        "X ^3 + X ^2 + X",
        "X ^4 + X ^3 + X ^2 + X",
        "X ^5 + X ^4 + X ^3 + X ^2 + X",
        "X * X ^5 + X ^5 + X ^4 + X ^3 + X ^2 + X",
        "sin ( X ^2 ) * cos ( X ) - 1",
        "sin ( X ) + sin ( X + X ^2 )",
        "log ( X + 1 ) + log ( X ^2 + 1 )",
        "sqrt ( X )"
    ]

    for i in range(len(ng_expressions)):
        tokens = ng_expressions[i].split(" ")
        expr_tree = tokens_to_tree(tokens, so)

        # Generating evaluation data matrix
        target, data, coords = generate_data(model, expr_tree)
        results_path = "../seeslab/test_03" + f"/nguyen{i}"
        clean_folder(results_path)

        for i in range(int(args.runs)):
            reg_data = mse_regression(model, data, max_iter=int(1e6), from_origin=True)
            reg_data.to_txt(results_path + f"/mse_run{i}.txt")
            reg_data.plot_all(results_path + f"/plots{i}")