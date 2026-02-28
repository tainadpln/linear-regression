# main.py

from src.train import train_model
from src.visualize import plot_regression

if __name__ == "__main__":
    x, y, final_w, final_b = train_model()
    print("Training completed")
    plot_regression(x, y, final_w, final_b)
    