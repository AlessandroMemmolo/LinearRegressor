import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GradientDescend(fun,gradient,x0, tol, alpha, maxIter):
    n = 0
    xpath = []
    xpath.append(x0)
    gpath = []
    gpath.append(gradient(x0))
    fpath = []
    fpath.append(fun(x0))
    while np.norm(gradient(x0)) > tol and n < maxIter:
        x0 = x0 - alpha*gradient(x0)
        n = n+1
        xpath.append(x0)
        gpath.append(gradient(x0))
        fpath.append(fun(x0))
    result = pd.DataFrame(data = [xpath,gpath,fpath], columns = ['x values', 'gradient values', 'function values'])
    return result
