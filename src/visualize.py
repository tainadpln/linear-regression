# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_regression(x, y, w, b):

    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = w * x_vals + b

    plt.scatter(x, y)
    plt.plot(x_vals, y_vals)

    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")

    plt.show()

    plt.savefig("reports/figures/regression_plot.png")