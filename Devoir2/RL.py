
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # pour RegressionLineaire


def PredictionRL(data):
    dataPair = []  # table pour 3c
    array = []

    for i in range(len(data.columns)):
        table = []
        for j in range(len(data.columns)):
            table.append(array)

        dataPair.append(table)

    tablePairCol = []
    for n in range(len(data.columns)):
        minDiff = 999999999999
        col1 = -1
        col2 = -1
        Y = data[data.columns[n]]
        pairCol = []
        for i in range(len(data.columns)-1):
            for j in range(i+1, len(data.columns)):
                if (i == n or j == n):
                    continue
                else:

                    X = data[[data.columns[i], data.columns[j]]]

                    regression = LinearRegression()
                    regression.fit(X, Y)
                    y_pred = regression.predict(X)

                    tableDiff = []
                    for k in range(len(y_pred)):
                        difference = np.absolute(
                            y_pred[k] - data.iloc[k, n])
                        tableDiff.append(difference)

                    moyenneDiff = np.mean(tableDiff)

                    dataPair[i][j].append(moyenneDiff)
                    if (moyenneDiff < minDiff):
                        minDiff = moyenneDiff
                        col1 = i
                        col2 = j

        pairCol.append(col1)
        pairCol.append(col2)
        tablePairCol.append(pairCol)
    return tablePairCol, dataPair
