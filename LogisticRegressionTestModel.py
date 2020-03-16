# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:54:48 2020

@author: Santosh Sah
"""
from LogisticRegressionUtils import (readLogisticRegressionXTest, readLogisticRegressionModel,
                                     saveLogisticRegressionYPred, readLogisticRegressionStandardScaler)

"""
test the model on testing dataset
"""
def testLogisticRegressionModel():
    
    X_test = readLogisticRegressionXTest()
    logisticRegressionStandardScaler = readLogisticRegressionStandardScaler()
    X_test = logisticRegressionStandardScaler.transform(X_test)
    
    logisticRegressionModel = readLogisticRegressionModel()
    
    y_pred = logisticRegressionModel.predict(X_test)
    saveLogisticRegressionYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testLogisticRegressionModel()