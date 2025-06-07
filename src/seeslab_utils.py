import matplotlib.pyplot as plt
import torch
import re
import os
import shutil

from tree import Node
from symbol_library import SymType

def distance(point1: torch.Tensor, point2: torch.Tensor) -> float:
    return torch.norm(point2 - point1).item()


def torch_to_coords(tensor: torch.Tensor) -> list:
    # Tensor compatibility check
    expected_shape = (1,1,None)
    for dim, expected_size in enumerate(expected_shape):
        if expected_size is not None and tensor.size(dim) != expected_size:
            raise ValueError(f"Expected shape {expected_shape}, but got {tensor.shape}")
    return tensor.flatten().tolist()


def expr_complexity (expr: Node, symbols: dict) -> int:
    """
    Computes the complexity of an expression tree.
    Complexity is defined as the number of INNER nodes in the tree.
    """
    if expr is None:
        return 0
    elif symbols[expr.symbol]["type"] not in [SymType.Operator, SymType.Fun]:
        return 0
    else:
        return 1 + expr_complexity(expr.left, symbols) + expr_complexity(expr.right, symbols)


def clean_folder(path):
    """
    Creats a new empty folder. If the folder already exists, it is overwritten.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)