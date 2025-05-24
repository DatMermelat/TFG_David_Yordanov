from argparse import ArgumentParser

import torch
import random
import os

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library
from seeslab_utils import torch_to_coords


def normal_sampling(model, points: int =100, path: str = None):
    # Sampling of points following 32 dimensional N(0,1) dist.
    for i in range(points):
        lrand = torch.randn(1,1,32)
        expr = model.decode(lrand)[0]

        tokens = expr.to_list("infix")
        expr_str = " ".join(tokens)

        with open(path, "a") as f:
            f.write(expr_str + "\n")

if __name__ == '__main__':
    parser = ArgumentParser(prog='Linear interpolation', description='Interpolate between two expressions')
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

    path = training_config["param_path"].replace("../params/","")
    path = path.replace("pt","txt")
    path = "ns_" + path
    path = os.path.join('../seeslab/samplings',path)

    points = 1000

    samp_data = normal_sampling(model, points, path)
    print("Sampling Complete!")