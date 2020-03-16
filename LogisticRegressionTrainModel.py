# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:55:10 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from LogisticRegressionUtils import (saveLogisticRegressionModel, readLogisticRegressionXTrain, readLogisticRegressionYTrain,
                                     saveLogisticLinearRegressionStandardScaler)

"""
Train simple linear regression model 
"""
def trainSimpleLinearRegressionModel():
    
    logisticRegressionStandardScalar = StandardScaler()
    
    X_train = readLogisticRegressionXTrain()
    y_train = readLogisticRegressionYTrain()
    
    logisticRegressionStandardScalar.fit(X_train)
    saveLogisticLinearRegressionStandardScaler(logisticRegressionStandardScalar)
    
    X_train = logisticRegressionStandardScalar.transform(X_train)
    
    logisticRegression = LogisticRegression(random_state = 1234)
    logisticRegression.fit(X_train, y_train)
    
    saveLogisticRegressionModel(logisticRegression)

if __name__ == "__main__":
    trainSimpleLinearRegressionModel()    
