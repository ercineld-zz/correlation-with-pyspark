import pandas as pd
import numpy as np
from pylab import scatter

#Import data set and check missing values
data = pd.read_csv('Icecream.csv',sep=',')
data.isnull().any()

#Assign numpy array to our choosen variables
ic = np.array(data['cons'])
temp = np.array(data['temp'])

#Scatter Plot
scatter(ic,temp)

#Correlation and covariance
np.cov(ic,temp)
np.corrcoef(ic,temp)

