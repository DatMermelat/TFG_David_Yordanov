from argparse import ArgumentParser

import torch
import random
import os

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library
from seeslab_utils import torch_to_coords, expr_is_unique

def sampling_to_txt (path: str,samp_data):
    with open(path,'w') as file:
        file.write(f"Unique expressions visited: {samp_data['n_uniq_expr']}\n")
        file.write("Point || Expression || Is Unique || Coordinates\n\n")

        for point_data in samp_data["point_data"]:
            file.write(f"{point_data['point']} || {point_data['expression']} || {point_data['norm']} || {point_data['is_unique']} || {point_data['coords']}\n")
        
def normal_sampling(model, points: int =100):
    origin = torch.zeros(1,1,32)
    coords = torch_to_coords(origin)
    n_uniq = 1
    
    # Initializing sampling dictionary
    samp_data = {
        "point_data": [],
        "unique_expressions": [],
        "n_uniq_expr": n_uniq
    }
    samp_data["point_data"].append({
        "point": "Origin",
        "expression": str(model.decode(origin)[0]),
        "norm": 0,
        "is_unique": True,
        "coords": coords
    })

    # Sampling of points following 32 dimensional N(0,1) dist.
    for point in range(points):
        lrand = torch.randn(1,1,32)
        expr = str(model.decode(lrand)[0])
        norm = torch.norm(lrand[0,0]).item()

        # Optional: Get latent sapce coordinates
        # Replace None with torch_to_coords(lrand)
        coords = None

        # Checking if the expression is visited for the first time
        if (expr_is_unique(expr, samp_data)):
            samp_data["unique_expressions"].append(expr)
            n_uniq += 1
            is_unique = True
        else:
            is_unique = False

        # Saving point data
        samp_data["point_data"].append({
            "point": point,
            "expression": expr,
            "is_unique": is_unique,
            "norm": norm,
            "coords": coords
        })
        
    samp_data["n_uniq_expr"] = n_uniq
    return samp_data
    
if __name__ == '__main__':
    parser = ArgumentParser(prog='Linear interpolation', description='Interpolate between two expressions')
    parser.add_argument("-config", default="../configs/custom_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    HVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"])

    path = training_config["param_path"].replace("../params/","")
    path = path.replace("pt","txt")
    path = "normal_" + path
    path = os.path.join('../seeslab/samplings',path)
    
    points = 1000
    
    samp_data = normal_sampling(model, points)
    sampling_to_txt(path, samp_data)
    print("Sampling Complete!")