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
      expressions = set()
      lines = file.readlines()
      for line in lines:
        expression = line.strip().split(" || ")[0]
        expressions.add(expression)
    return list(expressions)


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
  hvae_variables = "XYZABDEFGHIJKLMNOPQRSTUVW"

  for symbol in symbols:
    if symbol == "x":
      symbol_mapping[symbol] = "X"

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
  # aux = expression.translate(str.maketrans({"(": " ( ", ")": " ) "}))
  for char in expression:
      expression = expression.replace(char, f" {char} ")
  ws_expr = re.sub(r'\s+', ' ', expression).strip()
  return ws_expr


def convert_to_hvae(expressions: list, symbol_mapping: dict):
  """
  Converts BMS expressions to HVAE notation.
  INPUT: list of expression strings in BMS notation
  OUTPUT: list of expression strings in HVAE notation
  """
  converted = set()

  for expression in expressions:
      ws_expression = add_whitespaces(expression)
      tokens = ws_expression.split()

      for i, token in enumerate(tokens):
        if token in symbol_mapping:
          tokens[i] = symbol_mapping[token]

        # elif token == "-":
        #   del tokens[i]

      # Converting tokens back to string
      expression = " ".join(tokens)
      converted.add(expression)
  return list(converted)


if __name__ == '__main__':
  bms_dataset_filename = "ns_ng1_7.txt"
  hvae_dataset_filename = "ns_ng1_7.txt"
  bms_dataset_path = os.path.join("../", bms_dataset_filename)
  hvae_dataset_path = os.path.join("../", hvae_dataset_filename)
  reader = BmsReader()

  expressions = reader.extract_expressions(bms_dataset_path)
  symbols = get_symbols(expressions)
  symbol_mapping = create_symbol_mapping(symbols)
  hvae_expressions = convert_to_hvae(expressions, symbol_mapping)

  with open(hvae_dataset_path, 'w') as file:
    for expr in hvae_expressions:
      file.write(expr + "\n")
