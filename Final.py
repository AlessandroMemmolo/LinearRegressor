import LinReg as LR
import numpy as np
import pandas as pd
import DataGeneration as DG

data = DG.DataGenerator()
data.generate(1,0,np.linspace(0,10,100),2,2025)
data.plot()

mlr = LR.my_LinearRegressor()
mlr.train(data.x,data.y)
mlr.plot()