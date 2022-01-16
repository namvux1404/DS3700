from sklearn.naive_bayes import GaussianNB
import numpy as np


def prediction_bayes(data):
    dataPair = []  # table pour 3c
    array = []

    for i in range(len(data.columns)):
        table = []
        for j in range(len(data.columns)):
            table.append(array)

        dataPair.append(table)

    tablePairCol = []
    for n in range(len(data.columns)):
        maxAccuracy = 0
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

                    model = GaussianNB()
                    model.fit(X, Y)

                    y_pred = model.predict(X)

                    df = data[data.columns[n]].values

                    accuracy = np.sum(df == y_pred)/len(y_pred)
                    dataPair[i][j].append(accuracy)

                    if (accuracy >= maxAccuracy):
                        maxAccuracy = accuracy
                        col1 = i
                        col2 = j

        pairCol.append(col1)
        pairCol.append(col2)
        tablePairCol.append(pairCol)
    return tablePairCol, dataPair
