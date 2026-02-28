# src/train.py

import pandas as pd
from src.model import gradient_descent

def train_model():
    data = pd.read_csv("/workspaces/linear-regression/data/data.csv")

    x_train = data["YearsExperience"].values
    y_train = data["Salary"].values

    lr = 0.01
    iters = 10000

    final_w, final_b = gradient_descent(x_train, y_train, lr, iters)

    return x_train, y_train, final_w, final_b
