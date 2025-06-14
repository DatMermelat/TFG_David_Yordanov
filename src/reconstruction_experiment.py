from argparse import ArgumentParser

import os
import re
import torch

from model import HVAE
from hvae_utils import tokens_to_tree, create_batch, load_config_file
from symbol_library import generate_symbol_library
from tree import Node


class ReconstExperiment:

  def __init__(self, model, expressions: list, hvae_symbols: dict):
    self.model = model
    self.expressions = expressions
    self.symbols = hvae_symbols
    self.results = {}
    self.total_tries = 0
    self.total_succesful = 0

  def save_result(self, expression: str, decoded_expression: str, is_successful: bool):
    """
    Saves the result of the experiment.
    """
    if expression not in self.results:
      self.results[expression] = {"tries": 1,
                                  "n_successful": 1 if is_successful else 0,
                                  "decodings": {decoded_expression}}
    else:
      self.results[expression]["tries"] += 1
      self.results[expression]["decodings"].add(decoded_expression)
      if is_successful:
        self.results[expression]["n_successful"] += 1

  def run(self):
    """
    Experiment consisting of encoding expressions into
    the HVAE latent space and then decoding them back.
    """
    for expression in self.expressions:
      tokens = expression.split()
      if all(token in (list(self.symbols.keys()) + ["(",")"]) for token in tokens):
        tree = tokens_to_tree(tokens, self.symbols)

        # Encoding and decoding tree
        batch = create_batch([tree])
        vector = self.model.encode(batch)[0]
        decoded_expression = str(self.model.decode(vector)[0])
        self.total_tries += 1

        # Check if successful
        if decoded_expression == str(tree):
          self.save_result(str(tree), decoded_expression, is_successful=True)
          self.total_succesful += 1
        else:
          self.save_result(str(tree), decoded_expression, is_successful=False)

    # Sort results by number of successful tries in descending order
    self.results = dict(sorted(self.results.items(), key=lambda item: item[1]["n_successful"], reverse=True))

  def to_txt(self, path: str):
    with open(path, 'w') as file:
      file.write("total_tries: " + str(self.total_tries) + "\n")
      file.write("success_rate: " + str(self.total_succesful / self.total_tries) + "\n")
      for expr, result in self.results.items():
        file.write(f"{expr.strip()} || {result['tries']} || {result['n_successful']/result['tries']} || {result['decodings']}\n")


if __name__ == '__main__':
  parser = ArgumentParser(prog='Reconstruction experiment', description='Ecoding and decoding of expressions')
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

  dataset = "ng1_8.txt"
  dataset_path = os.path.join("../seeslab/reconstruction_experiment/datasets", dataset)

  expressions = []
  with open(dataset_path, 'r') as file:
    for line in file:
      expressions.append(line.strip())

  experiment = ReconstExperiment(model, expressions, so)
  experiment.run()
  experiment.to_txt(os.path.join("../seeslab/reconstruction_experiment", "results_" + dataset))