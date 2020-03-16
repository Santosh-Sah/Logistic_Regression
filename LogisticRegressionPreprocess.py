# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:50:35 2020

@author: Santosh Sah
"""

from LogisticRegressionUtils import (importLogisticRegressionDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importLogisticRegressionDataset("Logistic_Regression_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()