#!/usr/bin/env python
# coding: utf-8

# # JavaScript Stats

# In[2]:


# Importing all necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[3]:


# Importing Data through csv file
dataset = pd.read_csv("JavascriptDataset.csv")

# Parsing strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])


# In[4]:


# Import datetime and show top 5 values
from datetime import datetime
indexedDataset.head(5)


# In[5]:


indexedDataset.tail(5)


# In[69]:


# Plotting the Statistics
plt.xlabel("Month")
plt.ylabel("JavaScript")
plt.plot(indexedDataset)


# In[70]:


# determining rolling statistics 
# This calculating the unweighted mean and STD for the last n values
rolmean = indexedDataset.rolling(window=12).mean()

rolstd = indexedDataset.rolling(window=12).std()
print(rolmean, rolstd)

# the first 11 rules show 'NaN' because the first 11 averages are taken and given to the 12th Month value

# window of is 12 due to the data being represented Monthly


# In[71]:


#Plot rolling statistics
orig = plt.plot(indexedDataset, color = 'blue', label = 'Original')
mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show(block=False)


# In[72]:


#Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey-Fuller test:')

# given a lag of AIC (Akaike Information Criterion) 
# provides information about the exact values and actual values and analyzes the differences between them
dftest = adfuller(indexedDataset['JavaScript'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
    
print(dfoutput)


# In[73]:


# After DF Test, data is not stationary so we will...

# Estimate the Trend
# this is done by taking the log of the dataset
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)

# trend remians the sain but the value of y has been changed


# In[74]:


#Calculating Moving Average and Moving Standard Deviation

movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')


# In[75]:


#Calculating the difference between moving average and actual value 

datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NaN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[76]:


# Create a function that plots rolling statistics and performs  

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    # determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    # plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['JavaScript'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[77]:


# Use previously created function to evaluate 'differnece in moving average and actual value'

test_stationarity(datasetLogScaleMinusMovingAverage)


# In[78]:


# Calculate exponential decay weighted average to see the trend of the time series

exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[79]:


#Subtracting weighted average from the log scale

datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)


# In[80]:


# Shift values by a lag of 1

datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[81]:


# Drop NaN values
# result is somewhat stationary

datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[83]:


# Seeing the components of time series
# import seasonal decompose to view 3 components: trend, seasonality and residuals

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[85]:


#Check the noise for stationarity

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData )


# In[58]:


# ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

# Plotted pacf using Oridinary Least Squared Method
lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

# Plot ACF
# This calculates the value of 'q'

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y= -1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# Plot PACF
# This calculates the value of 'p'

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y= -1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[87]:


# Import function ARIMA
from statsmodels.tsa.arima_model import ARIMA

# AR Model

model = ARIMA(indexedDataset_logScale, order=(3,1,3))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')

# Calculating RSS (Residual Sum of Squares)
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues-datasetLogDiffShifting["JavaScript"])**2))
print('Plotting AR Model')


# In[108]:


#MA Model

model = ARIMA(indexedDataset_logScale, order=(1,1,0))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues-datasetLogDiffShifting["JavaScript"])**2))
print('Plotting MA Model')


# In[113]:


model = ARIMA(indexedDataset_logScale, order=(5,1,4))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["JavaScript"])**2))

#  AR RSS(0.0264) & MA RSS (0.0264) 
# ARIMA RSS is dropped to 0.0243


# In[114]:


# Converted the fitted values to a series format in order to generate predicitons

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[62]:


# Convert to cumulative sum
# Predictions for cumulative values

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[63]:


# Predictions for the fitted values

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['JavaScript'].iloc[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[31]:


# Exponention of whole data for it to come back to its original format

predictions_ARIMA= np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)


# In[64]:


# determining number of rows
# 211 months

indexedDataset_logScale


# In[33]:


#showing results for next 5 years (60 months)
# (211 + 60) = 271

results_ARIMA.plot_predict(1,271)


# In[115]:


# Showing future predictions in array format

results_ARIMA.forecast(steps=60)


# In[ ]:




