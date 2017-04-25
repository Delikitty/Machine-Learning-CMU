import os
import csv
import numpy as np
import LinReg
import LogReg


filepath = os.path.join('..','data')

(XTrain, yTrain, XTest, yTest)=LogReg.LogReg_ReadInputs(filepath)

LogReg.LogReg_SGA(XTrain, yTrain, XTest, yTest)

# (XTrain, yTrain, XTest, yTest)=LogReg.LogReg_ReadInputs(filepath)





