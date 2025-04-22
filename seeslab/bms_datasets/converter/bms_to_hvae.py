import os
import re


class BmsReader:

  def __init__(self):
    pass

  def extract_expressions(self, file_path):
    """
    Extracts only the mathematical expressions from the BMS dataset file.
    """
    with open(file_path, 'r') as file:
      expressions = []
      lines = file.readlines()
      for line in lines:
        expression = line.strip().split("||")[2]
        expressions.append(expression)
    return expressions


def get_symbols(expressions: list):
  """
  Returns the set of unique symbols appearing in the expressions.
  INPUT: list of expression strings
  OUTPUT: set of unique symbols (ignoring parentheses)
  """
  symbols = set()
  for expression in expressions:
    tokens = expression.translate(str.maketrans({"(": " ", ")": " "})).split()
    for token in tokens:
      if token not in symbols:
        symbols.add(token)
  return symbols


def create_symbol_mapping(symbols: set):
  """
  Returns a mapping of BMS symbols to HVAE notation.
  """
  symbol_mapping = {}
  for symbol in symbols:
    if symbol == "x":
      symbol_mapping[symbol] = "A"

    elif symbol == "**":
      symbol_mapping[symbol] = "^"

    elif symbol.startswith("pow"):
      symbol_mapping[symbol] = "^" + symbol[-1]

    elif symbol.startswith("_"):
      symbol_mapping[symbol] = "C"
  return symbol_mapping


def add_whitespaces(expression: str):
  """
  Adds whitespaces around every token in the expression.
  """
  aux = expression.translate(str.maketrans({"(": " ( ", ")": " ) "}))
  ws_expr = re.sub(r'\s+', ' ', aux).strip()
  return ws_expr


def convert_to_hvae(expressions: list, symbol_mapping: dict):
  """
  Converts BMS expressions to HVAE notation.
  INPUT: list of expression strings in BMS notation
  OUTPUT: list of expression strings in HVAE notation
  """
  converted = []

  for expression in expressions:
      ws_expression = add_whitespaces(expression)
      tokens = ws_expression.split()

      for i, token in enumerate(tokens):
        if token in symbol_mapping:
          tokens[i] = symbol_mapping[token]

        elif token == "-":
          del tokens[i]

      # Converting tokens back to string
      expression = " ".join(tokens)
      converted.append(expression)
  return converted


if __name__ == '__main__':
  bms_dataset_filename = "BMS_bmsprior.txt"
  hvae_dataset_filename = bms_dataset_filename.replace("BMS", "HVAE")
  path = "../seeslab/BMS"
  bms_dataset_path = os.path.join(path, bms_dataset_filename)
  hvae_dataset_path = os.path.join(path, hvae_dataset_filename)
  reader = BmsReader()

  expressions = reader.extract_expressions(bms_dataset_path)
  symbols = get_symbols(expressions)
  symbol_mapping = create_symbol_mapping(symbols)
  hvae_expressions = convert_to_hvae(expressions, symbol_mapping)

  with open(hvae_dataset_path, 'w') as file:
    for expr in hvae_expressions:
      file.write(expr + "\n")
