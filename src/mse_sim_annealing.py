from argparse import ArgumentParser

import numpy as np
import torch
import os
import copy

from model import HVAE
from hvae_utils import load_config_file, tokens_to_tree
from symbol_library import generate_symbol_library
from seeslab_utils import clean_folder
from evaluation import RustEval
from mse_regression import RegressionData, set_starting_point, get_expr_mse, generate_data


def cooling_function(temp: float, cooling_rate: float) -> float:
    return cooling_rate*temp


def mse_sim_annealing(model, data: np.array, params: dict) -> RegressionData:
    evaluator = RustEval(data, default_value = float('inf'), verbose = False)
    results = RegressionData()

    # Setting starting point
    current_vector, current_expr, current_mse = set_starting_point(model, evaluator, params["from_origin"])
    results.set_best(current_mse, current_expr, current_vector)
    best_mse, new_expr, new_vector = current_mse, current_expr, current_vector
    temp = float(params["start_temp"])

    # Safety Check (No division by 0, log of a negative number, etc.) at start
    if best_mse < float('inf'):
        results.save_expr_data(iteration=0, mse=current_mse, expr=current_expr, coords=current_vector)
        print(f"start: {current_expr}   mse: {current_mse}\n")
    else:
        if params["from_origin"]:
            raise ValueError("Invalid starting point. Restart the run with different independent values or set 'from_origin' to False.")
        else:
            raise ValueError("Invalid starting point. Restart run.")

    # Simulated Annealing
    i = 1
    while best_mse > float(params["tolerance"]) and i <= int(params["max_iter"]):
        # Taking random steps until there is a change in the decoded expression
        try:
            while str(new_expr) == str(current_expr):
                delta = float(params["step_size"]) * torch.randn(1,1,32)
                new_vector += delta
                new_expr = model.decode(new_vector)[0]
        except RecursionError:
            print(f"Recursion Limit: Stopped after {i} iterations.\n")
            return results

        # Evaluating expression and calculating mse
        new_mse = min(get_expr_mse(evaluator, new_expr))
        mse_delta = new_mse - current_mse

        # Applying SA probabilistic model for accepting changes
        if (mse_delta <= 0 or np.exp(-mse_delta / temp) > np.random.rand()) and new_expr.height() <= 7: # Accept change
            results.save_expr_data(iteration=i, mse=new_mse, expr=new_expr, coords=new_vector)
            current_mse, current_expr, current_vector = new_mse, new_expr, new_vector.clone()

            # Updating the best solution if necessary
            if current_mse <= best_mse:
                results.set_best(current_mse, current_expr, current_vector)
                best_mse = current_mse
        else: # Reject change and go back to previous solution
            new_expr, new_vector = current_expr, current_vector.clone()

            # Saving the best solution after the final iteration for plotting purposes
            if i == params["max_iter"]:
                results.save_expr_data(iteration=i, mse=current_mse, expr=current_expr, coords=current_vector)

        # Applying the cooling function after specified cooling delay
        if i % int(params["cooling_delay"]) == 0:
            temp = cooling_function(temp, params["cooling_rate"])
            print(f"Progress: {i/params['max_iter']*100: .2f}%", end='\r')
        i += 1

    # Exit messages
    if best_mse <= params["tolerance"]:
        print(f"MSE converged to tolerance {params['tolerance']} in {results.steps()} steps over {i-1} iterations.\n")
    else:
        print(f"Stopped after reaching max iterations ({int(params['max_iter'])}).\n")
    return results


if __name__ == '__main__':
    parser = ArgumentParser(prog='SA Regression', description='Symbolic regression by minimizing the MSE through an SA search')
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

    params = load_config_file("../configs/annealing_params.json")["simulated_annealing"]
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
    print(params)
    
    for i in range(0,len(ng_expressions)-3):
        tokens = ng_expressions[i].split(" ")
        expr_tree = tokens_to_tree(tokens, so)

        # Generating evaluation data matrix
        target, data, coords = generate_data(model, expr_tree)
        results_path = "../seeslab/sa_ng_t5e-3_01" + f"/nguyen{i}"
        clean_folder(results_path)

        for i in range(int(args.runs)):
            reg_data = mse_sim_annealing(model, data, params)
            reg_data.to_txt(results_path + f"/mse_run{i}.txt")
            reg_data.plot_all(results_path + f"/plots{i}")