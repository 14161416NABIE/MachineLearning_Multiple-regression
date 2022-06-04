import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv('FuelConsumption.csv')
print (df.head(4))
print(df.head(0))
cdf=df[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.tail(5))
print(cdf.values[:,1])
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
from sklearn import linear_model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[['ENGINESIZE','FUELCONSUMPTION_COMB','CYLINDERS']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
# Prediction
y_hat=regr.predict(test[['ENGINESIZE','FUELCONSUMPTION_COMB','CYLINDERS']])
x=np.asanyarray(test[['ENGINESIZE','FUELCONSUMPTION_COMB','CYLINDERS']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print('Residual sum of squares:%.2f' %np.mean((y_hat-y)**2))
print('variance score: %.2f' % regr.score(x,y))


