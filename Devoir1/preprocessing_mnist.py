# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# preprcessing_mnist.py
import csv
import numpy as np

# Pour les données d'entrainement
data = open('mnist_train.csv')
csv_file = csv.reader(data)

data_points_train = []

for row in csv_file:
    data_points_train.append(row)

# On enlève la première ligne, soit les "headers" de nos colonnes
data_points_train.pop(0)

data.close()

# Pour les données de test
data = open('mnist_test.csv')
csv_file = csv.reader(data)

data_points_test = []

for row in csv_file:
    data_points_test.append(row)

# On enlève la première ligne, soit les "headers" de nos colonnes
data_points_test.pop(0)

data.close()


# Prendre 500 data pour s'entrainer et 100 données de test pour accélerer le vitesse
data_points_reduit = [data_points_train[i] for i in range(500)]
data_test_reduit = [data_points_test[i] for i in range(100)]


# Convertir les données string en int pour travailler plus facilement
for i in range(len(data_points_reduit)):
    for j in range(0, 785):
        data_points_reduit[i][j] = int(data_points_reduit[i][j])

y_train = []  # Tableau des etiquettes

for row in data_points_reduit:
    y_train.append(row[0])

x_train = []  # Tableau de donnees

for row in data_points_reduit:
    x_train.append(row[1:785])

# Convertir les données string en int pour travailler plus facilement
for i in range(len(data_test_reduit)):
    for j in range(0, 785):
        data_test_reduit[i][j] = int(data_test_reduit[i][j])

y_test = []  # Tableau des etiquettes

for row in data_test_reduit:
    y_test.append(row[0])

x_test = []  # Tableau de donnees

for row in data_test_reduit:
    x_test.append(row[1:785])

# Convertir image de train en densite noir et blanc pour faiciliter la classification
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        if x_train[i][j] != 0:
            # Nous devons diviser par 255.0 et non 255 pour convertir ces int en float
            x_train[i][j] = round(int(x_train[i][j]) / 255.0)


# Convertir image de test en densite noir et blanc pour faiciliter la classification
for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if x_test[i][j] != 0:
            # Nous devons diviser par 255.0 et non 255 pour convertir ces int en float
            x_test[i][j] = round(int(x_test[i][j]) / 255.0)


def import_data():  # Fonction pour exporter les données
    return x_train, y_train, x_test, y_test
