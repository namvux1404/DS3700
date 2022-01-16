
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression  # pour RegressionLineaire
from sklearn.naive_bayes import GaussianNB
from RL import PredictionRL
from prediction_bayes import prediction_bayes
from sklearn.decomposition import PCA
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork

# pour indiquer a pandas que le tableau est indexe par pays
dataA = pd.read_csv("dataA.csv", index_col=0)

print('head :')
print(dataA.head())

print('shape = ')
print(dataA.shape)

print(dataA.dtypes)


def verifyDataType():  # iteration colum -- test type data
    for column in dataA.columns:
        if (dataA[column].dtypes == "float64"):
            continue
        else:
            print('Data of this column not appropriate : ' + column)
            dataA[column] = dataA[column].astype(float)

    print("--- done Verify --")


def countMissingValue(data):  # test si le nombre de valeur manquante est < 12
    bool = False
    table = data.isnull().sum(axis=1)
    # print(len(table))
    for i in range(len(table)):
        if(table.iloc[i, ] < 12):
            bool = True
            continue
        else:
            bool = False

    return bool


def printColumn(data):  # pour impimer les colonnes de dataFrame
    for column in data.columns:
        print(data[column])


def updateNaNvaluebyMedian(indexNaN, arrayMediane):  # update data avec median
    for i in range(len(indexNaN)):  # index column
        if (len(indexNaN[i]) > 0):
            for j in range(len(indexNaN[i])):  # index row = indexNaN[i][j]
                dataA.iloc[indexNaN[i][j], i] = arrayMediane.iloc[i, ]


def updateRL():  # update valeur manquante avec RL
    for i in range(len(indexValeurManquante)):  # index column
        if (len(indexValeurManquante[i]) > 0):

            listCountry = []  # liste des pays avec valeurs manquantes a remplir
            for j in range(len(indexValeurManquante[i])):
                indexRow = indexValeurManquante[i][j]
                country = dataA.index[indexRow]
                listCountry.append(country)

            columnLabel = dataA.columns[i]  # Label pour colonne a predire
            data = dataA.drop(listCountry)

            Y = data[columnLabel]
            X = data.loc[:, data.columns != columnLabel]

            regression = LinearRegression()
            regression.fit(X, Y)

            for j in range(len(indexValeurManquante[i])):
                indexRow = indexValeurManquante[i][j]
                country = dataA.index[indexRow]
                y_pred = regression.predict(
                    np.array([dataA.loc[country, dataA.columns != columnLabel]]))

                dataA.iloc[indexRow, i] = y_pred


def normaliserData(data, TabMean, TabStd):  # fonction pour normaliser les donnes
    table = data.copy()
    for i in range(len(table.columns)):
        mean = TabMean.iloc[i, ]  # moyenne de la colonne
        std = TabStd.iloc[i, ]  # ecart type de la colonne
        for j in range(len(table)):
            value = (table.iloc[j, i] - mean) / std  # normaliser
            table.iloc[j, i] = value                 # update valeur

    return table


def exportToCsv(data, filename):  # fonction pour exporter en CSV
    data.to_csv(filename)


def exportToJSON(array, filename):  # fonction pour exporter en Json
    with open(filename, 'w') as f:
        json.dump(array, f)


verifyDataType()
# print(dataA.dtypes)
exportToCsv(dataA, 'dataA.csv')
print("Tous les pays ont moins de 12 valeurs manquantes: " +
      str(countMissingValue(dataA)))
# -------------------
indexValeurManquante = []  # index pour chaque colonne

for i in range(len(dataA.columns)):  # save index for NaN Value
    arrayIndex = []
    for j in range(len(dataA)):
        if (np.isnan(dataA.iloc[j, i])):
            arrayIndex.append(j)

    indexValeurManquante.append(arrayIndex)

# print(indexValeurManquante)

# -------------------- pour Mediane et valeur
arrayMedianColumn = dataA.median()  # mediane pour chaque colonne

print(arrayMedianColumn)  # format column in row --> iloc(i,)

print('------ Before ajust value NaN ---------')


# update avec la mediane
updateNaNvaluebyMedian(indexValeurManquante, arrayMedianColumn)
print('------ After ajust value NaN ---------')

# -------------- fin 1er update --------

#### ------- 2e update avec RL  multiple -------- ####

updateRL()  # 1er fois
print(' ------ 1er fois RL')
# printColumn(dataA)

updateRL()  # 2e fois
print(' ------- 2e fois RL')
# printColumn(dataA)  # ----> ok

print(dataA.dtypes)
exportToCsv(dataA, 'dataB.csv')  # ---> sauvergarde en csv dataB.csv --- ok

# Il faut update Mediane aussi
arrayMedianColumn = dataA.median()  # updated median after 2 RL
dataC = dataA.copy()  # for C

for i in range(len(dataC.columns)):  # Creation tableC

    mediane = arrayMedianColumn.iloc[i, ]
    for j in range(len(dataC)):
        if (dataC.iloc[j, i] > mediane):

            dataC.iloc[j, i] = 1
        else:
            dataC.iloc[j, i] = 0

# print(dataC)
exportToCsv(dataC, 'dataC.csv')  # ---> sauvergarde en csv dataC.csv  --- ok
print(' ##### ------- Fin etape 1 ----------- ######')
### --------- 2 . Correlation -------- #####

print(' ##### ------- Etape 2 ----------- ######')


# Fonction pour calculer la correlation entre les colonnes
def calculerCorrelation(data):
    tableCorr = []
    for column1 in dataA.columns:
        rowCorr = []
        for column2 in dataA.columns:
            if(column1 == column2):  # dans le cas ou c'est le meme colonne
                rowCorr.append(1)  # met a 0 car on cherche plus forte
            else:
                corr = data[column1].corr(data[column2])
                rowCorr.append(corr)

        tableCorr.append(rowCorr)

    return tableCorr


def indexMaxCorr(tableCorr):  # fonction pour sortir index de corr Max de chaque colonne
    tabIndexMaxCor = []
    for i in range(len(tableCorr)):
        array = tableCorr[i].copy()
        array[i] = 0  # remove la correlation avec lui meme
        indexMax = np.argmax(np.absolute(array))
        tabIndexMaxCor.append(indexMax.item())

    return tabIndexMaxCor


def ordreCorr(tableCorr):
    moyenneCorr = []
    for array in tableCorr:
        meanCorr = np.mean(np.absolute(array))
        moyenneCorr.append(meanCorr)

    # print(moyenneCorr)
    sortedIndex = np.flip(np.argsort(moyenneCorr))
    return sortedIndex.tolist()


tableCorr = calculerCorrelation(dataA)
indexMaxCor = indexMaxCorr(tableCorr)
sortedIndex = ordreCorr(tableCorr)
# print(sortedIndex)
exportToJSON(tableCorr, 'corr.json')  # ----> ok
exportToJSON(indexMaxCor, 'max_corr.json')  # ------> ok
exportToJSON(sortedIndex, 'ordre.json')  # ------> ok
# print(tableCorr)

print(' ##### ------- Fin etape 2 ----------- ######')
print(' ##### ------- Etape 3 ----------- ######')

arrayMeanColumn = dataA.mean()
arrayEcartTypeColumn = dataA.std()
# print(arrayEcartTypeColumn)


NormalDataA = normaliserData(dataA, arrayMeanColumn, arrayEcartTypeColumn)
# printColumn(NormalDataA)

# ------- Prediction avec RL -------------


def determinePairCol(data):
    minMoyenne = 999999999999999
    col1 = -1
    col2 = -1
    for i in range(len(data)):
        for j in range(len(data[i])):
            if (i != j):
                mean = np.mean(data[i][j])
                if (mean < minMoyenne):
                    minMoyenne = mean
                    col1 = i
                    col2 = j

    return [dataA.columns[col1], dataA.columns[col2]]


# commentaire pour le moment
tablePairCol, dataPair = PredictionRL(NormalDataA)  # -- okkk
pairColonne = determinePairCol(dataPair)
# print(pairColonne)
# print(dataPair)
exportToJSON(tablePairCol, 'lineaire_paires_colonnes.json')  # ------> ok
# pairCol(dataA)


# ------- Prediction avec Classifieur bayesien -------------
def determinePairCol_Bayes(data):
    maxAccuracy = 0
    col1 = -1
    col2 = -1
    for i in range(len(data)):
        for j in range(len(data[i])):
            if (i != j):

                meanAccuracy = np.mean(data[i][j])
                # print(meanAccuracy)
                if (meanAccuracy > maxAccuracy):
                    maxAccuracy = meanAccuracy
                    col1 = i
                    col2 = j

    return [dataA.columns[col1], dataA.columns[col2]]


tablePairCol_bayes, dataPairBayes = prediction_bayes(dataC)
# print(dataPairBayes[0])
pairColonneBayes = determinePairCol_Bayes(dataPairBayes)
# print(pairColonneBayes)
#print('moyenneDiff avec C14 et C18 = ' + str(moyenneDiff))
exportToJSON(tablePairCol_bayes, 'bayesien_paires_colonnes.json')  # ------> ok
# print(tablePairCol_bayes)


print(' ##### ------- Etape 4 : Reduit dimensionnelle ----------- ######')


def pca(dim, data):
    pca = PCA(n_components=dim)
    x_pca = pca.fit_transform(data)
    return x_pca
    # print(x_pca.shape)


def plot(data, x_pca):
    label = data.index.values
    x = x_pca[:, 0]
    y = x_pca[:, 1]
    print(label)
    plt.figure(figsize=(15, 8))
    plt.scatter(x, y, c='green')
    plt.xlabel('First principle component')
    plt.ylabel('Second principle component')
    for i, txt in enumerate(label):
        plt.annotate(txt, (x[i], y[i]))
    plt.show()


x_pca = pca(2, NormalDataA)
plot(NormalDataA, x_pca)


x1_pca = pca(5, NormalDataA)
# print(x1_pca.shape)
# print(x1_pca)

print(' ##### ------- Etape 5 : Reseau bayesien ----------- ######')
#nbFleches = 100
#limit_parent = 4
#gdp_node = Node(dataA.columns[0], name='GDP')
#internet_node = Node(dataA.columns[1], name='Internet')
#bayesnet = BayesianNetwork("Devoir2")
# bayesnet.add_states(
#    gdp_node,
#    internet_node
# )
#bayesnet.add_edge(gdp_node, internet_node)
# bayesnet.bake()
# with open('reseau.json', 'w') as f:
#    f.write(bayesnet.to_json())
