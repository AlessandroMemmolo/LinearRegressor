import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def DataGenerator(m,b,x,sigma,seed):
    np.random.seed(seed)
    y = np.random.normal(m*x + b,sigma)

    data = pd.DataFrame(y,x)
    data.to_csv("c:/Users/aless/OneDrive/Desktop/Projects/LinearRegressor/data.csv")

    