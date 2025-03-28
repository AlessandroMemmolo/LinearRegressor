import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataGenerator():

    def generate(self,m,b,x,sigma,seed):
        np.random.seed(seed)
        self.m = m
        self.b = b
        self.sigma = sigma
        self.x = x
        self.y = np.random.normal(m*x + b,sigma)

    def save(self,namePath):
        temp = pd.DataFrame({'x':self.x,'y':self.y})
        temp.to_csv(namePath) 
    
    def plot(self):
        plt.plot(self.x,self.y,'ro')
        plt.plot(self.x,self.b + self.m*self.x)
        plt.show()

    