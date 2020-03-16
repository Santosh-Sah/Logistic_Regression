# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:50:05 2020

@author: Santosh Sah
"""

import pandas as pd
from LogisticRegressionUtils import readLogisticRegressionModel, readLogisticRegressionStandardScaler

def predict():
    
    logisticRegression = readLogisticRegressionModel()
    logisticRegressionStandardScaler = readLogisticRegressionStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(logisticRegressionStandardScaler.transform(inputValue))
    
    predictedValue = logisticRegression.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()