# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Travailler sur datasets : MNIST
# Fichier main pour generer les algorithmes
# et afficher les resultats pour evaluer la performance

import numpy as np
import pandas as pd
from mnist_similarity import mnist_import_data

from knn import knn_processing
from kmedoids import kmedoids_processing
from isomap import isomap_processing
from pcoa import pcoa_processing
from partitionBinaire import PB_processing


# fonction pour calculer l'exactitude de l'algorithme
def accuracy_mnist(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    succes = np.sum(a == b)
    return round(succes/len(y_predict), 5)


print('###------ Dataset MNIST ------####')
print("---importing data.....")
mnistX_train, mnistY_train, mnistX_test, mnistY_test, mnistD_train, mnistD_test, mnistDE_train, mnistDE_test = mnist_import_data()
print('# Data train = ' + str(len(mnistX_train)))
print('# Data test = ' + str(len(mnistX_test)))
# print(np.shape(mnistD_train))
# print(np.shape(mnistD_test))
# print(np.shape(mnistDE_train))
# print(np.shape(mnistDE_test))
print("- imported ")
print('\n')

# -----------------------------------------------------
print('---------- 1) ALGORITHME KNN ----------')
print('... Processing ....')

knn_mnist = knn_processing(mnistD_train, mnistD_test,
                           mnistY_train, mnistY_test)

print('----')
print('-- Avec la distance euclidienne --')
knn_mnist_eulidien = knn_processing(mnistDE_train, mnistDE_test,
                                    mnistY_train, mnistY_test)
print('\n')

# -----------------------------------------------------
print('---------- 2) ALGORITHME K-medoids ----------')
print('... Processing ....')
kmedoids_mnist = 10
print('Execute avec ' + str(kmedoids_mnist)+'-medoids')

kmedoids_mnist_predict = kmedoids_processing(
    mnistX_train, mnistY_train, mnistD_train, mnistD_test, kmedoids_mnist)

print(' -> done processing')
print('-- Etape evaluation : ')

score_kmedoids = accuracy_mnist(kmedoids_mnist_predict, mnistY_test)
print('Accuracy = ' + str(score_kmedoids))
print('----')

print('-- Avec la distance euclidienne --')
kmedoids_mnist_predict_euclidien = kmedoids_processing(
    mnistX_train, mnistY_train, mnistDE_train, mnistDE_test, kmedoids_mnist)
print(' -> done processing')
print('-- Etape evaluation : ')

score_kmedoids_euclidien = accuracy_mnist(
    kmedoids_mnist_predict_euclidien, mnistY_test)
print('Accuracy = ' + str(score_kmedoids_euclidien))
print('----')
print('\n')

# -----------------------------------------------------
print('---------- 3) ALGORITHME Isomap + application KNN ----------')
print('... Processing isomap ....')
n_comp_isomap = 8
print('Reduit dimentionnel à ' + str(n_comp_isomap))

isomap_mnist = isomap_processing(
    mnistD_train, mnistD_test, mnistY_train, mnistY_test, n_comp_isomap)

print('----')
print('-- Avec la distance euclidienne --')
isomap_mnist = isomap_processing(
    mnistDE_train, mnistDE_test, mnistY_train, mnistY_test, n_comp_isomap)
print('\n')

# -----------------------------------------------------
print('---------- 4) ALGORITHME PCoA + application KNN ----------')
print('... Processing PCoA ....')

n_comp_pcoa = 8
print('Reduit dimentionnel à ' + str(n_comp_pcoa))

pcoa_mnist = pcoa_processing(
    mnistD_train, mnistD_test, mnistY_train, mnistY_test, n_comp_pcoa)

print('----')
print('-- Avec la distance euclidienne --')
pcoa_mnist = pcoa_processing(
    mnistDE_train, mnistDE_test, mnistY_train, mnistY_test, n_comp_pcoa)
print('\n')


# -----------------------------------------------------
print('---------- 5) ALGORITHME Partition Binaire ----------')
print('... Processing ....')
cluster_mnist = 30  # nombre de cluster pour PA
print('Execute avec ' + str(cluster_mnist)+' partitions')

pb_mnist_predict = PB_processing(
    mnistD_train, mnistD_test, mnistY_train, cluster_mnist)

score_partitionBinaire = accuracy_mnist(pb_mnist_predict, mnistY_test)
print('Accuracy = ' + str(score_partitionBinaire))
print('----')

print('-- Avec la distance euclidienne --')
pb_mnist_predict_euclidien = PB_processing(
    mnistDE_train, mnistDE_test, mnistY_train, cluster_mnist)

score_pb_euclidien = accuracy_mnist(
    pb_mnist_predict_euclidien, mnistY_test)
print('Accuracy = ' + str(score_pb_euclidien))
print('\n')
print('-----------------------------------------------------')
