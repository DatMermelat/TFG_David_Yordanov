from argparse import ArgumentParser
from typing import Dict, Any

import numpy as np
import torch
import os

from model import HVAE
from hvae_utils import load_config_file, tokens_to_tree, create_batch
from symbol_library import generate_symbol_library
from evaluation import RustEval
from tree import Node

if __name__ == '__main__':
    parser = ArgumentParser(prog='Evaluation Test', description='Testing expression evaluation using RustEval')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"], weights_only = False)

    str_expr = "X ^2 + C ^3 + 1"
    tokens_expr = str_expr.split()
    treeA = tokens_to_tree(tokens_expr, so)
    BtreeA = create_batch([treeA])
    vector = model.encode(BtreeA)[0]
    decoded_expr = model.decode(vector)[0]
    print(decoded_expr)