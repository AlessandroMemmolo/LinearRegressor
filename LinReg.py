import GradientDescend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class my_LinearRegressor():


    def train(self,x,y):
        
        #x is a np array
        shap = np.shape(x)
        self.numSamples = shap[0]
        if x.size == shap[0]:
            self.features = 1
        else:
            self.features = shap[1]
        self.beta = np.zeros((1,self.features))

        self.xdata = np.concat((np.ones((self.numSamples,1)),x))
        self.ydata = y

        def MSE_xy(self,beta):
            return(np.sum((beta*self.xdata-self.ydata)**2))
        
        def MSE_xy_grad(self,beta):
            return(2*np.transpose(self.xdata)*(beta*self.xdata-self.ydata))
        
        self.results = GradientDescend.GradientDescend(MSE_xy,MSE_xy_grad,self.beta,0.001,1,10000)
        self.beta = self.results.iat[-1,0]
        self.MSE = self.results.iat[-1,2]

    def predict(self,x):
        return self.beta[0] + self.beta[1:]*x
    
    def plot(self):
        plt.plot(self.xdata,self.y,'ro')
        plt.plot(self.xdata,self.beta*self.xdata)
        plt.show()