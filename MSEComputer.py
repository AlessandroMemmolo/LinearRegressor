import numpy as np

def MSE(yPred,yTrue):
    return(np.sum((yPred-yTrue)**2))