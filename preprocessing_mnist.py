# preprcessing_mnist.py
import csv
import numpy as np
import matplotlib.pyplot as plt

data = open('mnist_train.csv')
csv_file = csv.reader(data)

data_points_train = []

for row in csv_file:
    data_points_train.append(row)

# On enlève la première ligne, soit les "headers" de nos colonnes
data_points_train.pop(0)

data.close()

data = open('mnist_test.csv')
csv_file = csv.reader(data)

data_points_test = []

for row in csv_file:
    data_points_test.append(row)

# On enlève la première ligne, soit les "headers" de nos colonnes
data_points_test.pop(0)

data.close()


# Prendre 10000 data pour tester pour accélerer le vitesse
data_points_reduit = [data_points_train[i] for i in range(500)]
data_test_reduit = [data_points_test[i] for i in range(100)]
# print('# Data train = ' + str(len(data_points_reduit)))
# print('# Data test = ' + str(len(data_test_reduit)))

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

# print(x_test[0])
# matrix = np.reshape(x_test[0], (28, 28))
# print(matrix)

# Convertir image en densite noir et blanc pour faiciliter la classification
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        if x_train[i][j] != 0:
            # Nous devons diviser par 255.0 et non 255 pour convertir ces int en float
            x_train[i][j] = round(int(x_train[i][j]) / 255.0)

# print(x_train[0])
# Convertir image en densite noir et blanc pour faiciliter la classification
for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if x_test[i][j] != 0:
            # Nous devons diviser par 255.0 et non 255 pour convertir ces int en float
            x_test[i][j] = round(int(x_test[i][j]) / 255.0)


def import_data():
    return x_train, y_train, x_test, y_test


# A = [7, 6, 1, 7, 4, 1, 9, 1, 4, 1, 9, 1, 4,
 #    7, 1, 1, 4, 7, 1, 1, 1, 1, 1, 1, 1, 7, 1, 4,
 #    0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 4, 4, 1, 6, 1, 1, 1, 0, 4, 1, 7, 9, 1, 4, 1, 1, 2, 1, 9, 4, 3, 1, 7, 0, 9, 2, 1, 9, 1, 9, 2, 7, 1, 1, 2, 1, 2, 4,
 #    1, 1, 4, 9, 3, 1, 2, 3, 1, 4, 1, 1, 9, 1]
# B = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2, 4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3, 7, 4, 6, 4, 3, 0, 7, 0, 2, 9, 1, 7, 3, 2, 9, 7, 7, 6, 2, 7, 8, 4,
 #    7, 3, 6, 1, 3, 6, 9, 3, 1, 4, 1, 7, 6, 9]

# a = np.array(A)
# b = np.array(B)
# A = [1, 1, 1, 2, 3, 4, 5, 5, 1, 2, 3]
# most_frequent_class = np.bincount(A).argmax()
# succes = np.sum(a == b)
# print(succes)
# print("----- Example Image ------")
# Conversion de notre vecteur d'une dimension en 2 dimensions
# matrix = np.reshape(x_train[0], (28, 28))

# plt.imshow(matrix, cmap='gray')

# Affiche un 5, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()
