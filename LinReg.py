import GradientDescend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class my_LinearRegressor():


    def train(self,x,y):
        
        #x is a np array
        self.features = np.size(x,2)
        self.beta = np.zeros(1,self.features)
        self.numSamples = np.size(x,1)

        self.xdata = np.concat(np.ones(self.numSamples,1),x)

        def MSE_xy(self,beta,y):
            return(np.sum((beta*self.xdata-y)**2))
        
        def MSE_xy_grad(self,beta,y):
            return(2*np.transpose(self.xdata)*(beta*self.xdata-y))
        
        self.results = GradientDescend.GradientDescend(MSE_xy,MSE_xy_grad,self.beta,0.001,1,10000)
        self.beta = self.results.iat[-1,0]
        self.MSE = self.results.iat[-1,2]

    def predict(self,x):
        return self.beta*x