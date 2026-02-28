# src/model.py

import numpy as np

def cost_function(x, y, w, b):
    m = len(x)
    cost_sum = 0

    for i in range(m):
        f = w * x[i] + b
        cost = (f - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def gradient_function(x, y, w, b):
    m = len(x)
    dc_dw = 0
    dc_db = 0

    for i in range(m):
        f = w * x[i] + b

        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])

    dc_dw = (1 / m) * dc_dw
    dc_db = (1 / m) * dc_db

    return dc_dw, dc_db

def gradient_descent(x, y, lr, iters):
    w = 0
    b = 0

    for i in range(iters):
        dc_dw, dc_db = gradient_function(x, y, w, b)

        w = w - lr * dc_dw
        b = b - lr * dc_db

        print(f"Iteration {i}: Cost{cost_function(x, y, w, b)}")

    return w, b
