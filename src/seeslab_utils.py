from typing import Dict, Any

import matplotlib.pyplot as plt
import torch
import re
import os
import shutil

def distance(point1: torch.Tensor, point2: torch.Tensor) -> float:
    return torch.norm(point2 - point1).item()


def torch_to_coords(tensor: torch.Tensor) -> list:
    # Tensor compatibility check
    expected_shape = (1,1,None)
    for dim, expected_size in enumerate(expected_shape):
        if expected_size is not None and tensor.size(dim) != expected_size:
            raise ValueError(f"Expected shape {expected_shape}, but got {tensor.shape}")
    return tensor.flatten().tolist()


def expr_complexity (expr: str) -> int:
    symbols = ["+", "-", "*", "/", "^", "sin", "cos", "exp", "sqrt", "log"]
    complexity = 0

    for symbol in symbols:
        if len(symbol) == 1:  # Single-character operator
            complexity += expr.count(symbol)
        else:  # Multi-character function
            complexity += len(re.findall(rf"\b{symbol}\b", expr))

    return complexity


# Method that creates an empty folder. If the folder already exists, it's overwritten.
def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)