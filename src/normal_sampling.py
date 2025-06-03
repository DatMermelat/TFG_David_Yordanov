from argparse import ArgumentParser

import torch
import random
import os

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library
from seeslab_utils import clean_folder, expr_complexity
from plot_utils import generate_plots

class NormalSampling:

    def __init__(self, model, symbols: dict, points: int = 1000, std_dev: float = 1.0):
        self.model = model
        self.symbols = symbols
        self.points = points
        self.std_dev = std_dev
        self.samples = []


    def generate_plot_data(self):
        """
        Generates plots for the sampled expressions.
        Plots:
        - Expression complexity vs modulus in latent space
        - Expression complexity histogram
        """
        self.plots = []

        # Complexity vs Modulus
        complexities = [sample["complexity"] for sample in self.samples]
        norms = [sample["norm"] for sample in self.samples]
        self.plots.append({
            "type": "scatter",
            "x": norms,
            "y": complexities,
            "title": "Complexity vs Modulus",
            "xlabel": "Modulus in Latent Space",
            "ylabel": "Decoded Expression Complexity"
        })

        # Complexity Histogram
        self.plots.append({
            "type": "histogram",
            "data": complexities,
            "title": "Complexity Histogram",
            "xlabel": "Complexity",
            "ylabel": "Relative Frequency",
            "bins": "auto"
        })


    def expr_frequencies(self):
        """
        Calculates the frequency of each expression in the samples.
        Returns a dictionary with expressions as keys and their frequencies as values.
        """
        expr_freq = {}
        for sample in self.samples:
            expr = str(sample["expr"])
            if expr not in expr_freq:
                expr_freq[expr] = 0
            expr_freq[expr] += 1
        sorted_expr_freq = sorted(expr_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_expr_freq


    def run(self):
        """
        Samples points in the latent space using a normal distribution.
        """
        for i in range(self.points):
            lrand = self.std_dev * torch.randn(1,1,32)
            expr = self.model.decode(lrand)[0]
            complexity = expr_complexity(expr, self.symbols)

            self.samples.append({
                "expr": expr,
                "complexity": complexity,
                "norm": lrand.norm().item()
            })

        self.generate_plot_data()


if __name__ == '__main__':
    parser = ArgumentParser(prog='Normal Sampling', description='Sample expressions from a normal distribution')
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
    points = 10000
    scatters = {"type": "scatter",
                "x": [],
                "y": [],
                "title": "Complexity vs Modulus (accumulated)",
                "xlabel": "Modulus in Latent Space",
                "ylabel": "Decoded Expression Complexity"
                }

    for i in range(4):
        std_dev = (i+1)/2

        sampler = NormalSampling(model, so, points, std_dev)
        sampler.run()

        path = f"../seeslab/samplings/{std_dev}_{points}_bms_ng1_8"
        clean_folder(path)

        scatters["x"] += sampler.plots[0]["x"]
        scatters["y"] += sampler.plots[0]["y"]
        generate_plots(sampler.plots, path)
        generate_plots([scatters], path)
        expr_freq = sampler.expr_frequencies()

        with open(os.path.join(path, "samples.txt"), "w") as f:
            for (expr, freq) in expr_freq:
                f.write(f"{expr} || {freq}\n")


        print("Sampling Complete! Results saved to:", path)