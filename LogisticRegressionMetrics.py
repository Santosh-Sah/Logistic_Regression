# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from LogisticRegressionUtils import (readLogisticRegressionYTest, readLogisticRegressionYPred)

"""

calculating logistic regression confussion matrix

"""
def testLogisticRegressionConfussionMatrix():
    
    y_test = readLogisticRegressionYTest()
    y_pred = readLogisticRegressionYPred()
    
    logisticRegressionConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(logisticRegressionConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[56  2]
    [ 5 17]]
    
    """
"""
calculating accuracy score

"""

def testLogisticRegressionAccuracy():
    
    y_test = readLogisticRegressionYTest()
    y_pred = readLogisticRegressionYPred()
    
    logisticRegressionConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(logisticRegressionConfussionAccuracy) #.9125%

"""
calculating classification report

"""

def testLogisticRegressionClassificationReport():
    
    y_test = readLogisticRegressionYTest()
    y_pred = readLogisticRegressionYPred()
    
    logisticRegressionConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(logisticRegressionConfussionClassificationReport)
    
    """
                 precision    recall  f1-score   support

          0       0.92      0.97      0.94        58
          1       0.89      0.77      0.83        22

avg / total       0.91      0.91      0.91        80

    """
    
if __name__ == "__main__":
    #testLogisticRegressionConfussionMatrix()
    #testLogisticRegressionAccuracy()
    testLogisticRegressionClassificationReport()