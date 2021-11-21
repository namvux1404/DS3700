# Xuan - Matricule
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Travailler sur 2 datasets : MNIST et Adults
# Fichier main pour generer les algorithmes
# et afficher les resultats pour evaluer la performance

import numpy as np
import pandas as pd
from mnist_similarity import mnist_import_data
from adult_similarity import adult_import_data

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
mnistX_train, mnistY_train, mnistX_test, mnistY_test, mnistD_train, mnistD_test = mnist_import_data()
adultX_train, adultY_train, adultX_test, adultY_test, adultD_train, adultD_test = adult_import_data()

print('---------- 1) Algorithme KNN ----------')
print('... Processing ....')
kneighbors_mnist = 1
print('Execute avec ' + str(kneighbors_mnist)+'-NN')
knn_mnist = knn_processing(mnistD_train, mnistD_test,
                           mnistY_train, kneighbors_mnist)
print(' -> done processing')
print('-- Etape evaluation : ')

score_knn = accuracy_mnist(knn_mnist, mnistY_test)
print('Accuracy = ' + str(score_knn))
print('----')
print('\n')

print('---------- 2) Algorithme K-medoids ----------')
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
print('\n')

print('---------- 3) Algorithme Isomap + application KNN ----------')
print('... Processing isomap ....')
n_comp_isomap = 8
print('Reduit dimentionnel à ' + str(n_comp_isomap))

isomap_mnist = isomap_processing(
    mnistD_train, mnistD_test, mnistY_train, mnistY_test, n_comp_isomap)

print('----')
print('\n')

print('---------- 4) Algorithme PCoA + application KNN ----------')
print('... Processing isomap ....')

n_comp_pcoa = 8
print('Reduit dimentionnel à ' + str(n_comp_pcoa))

pcoa_mnist = pcoa_processing(
    mnistD_train, mnistD_test, mnistY_train, mnistY_test, n_comp_pcoa)

print('----')
print('\n')

print('---------- 5) Algorithme Partition Binaire ----------')
print('... Processing ....')
cluster_mnist = 30
print('Execute avec ' + str(cluster_mnist)+' partitions')
pb_mnist_predict = PB_processing(
    mnistD_train, mnistD_test, mnistY_train, cluster_mnist)

#print(' -> done processing')
#print('-- Etape evaluation : ')
# print(knn_mnist)
score_partitionBinaire = accuracy_mnist(pb_mnist_predict, mnistY_test)
print('Accuracy = ' + str(score_partitionBinaire))
print('----')
print('\n')
print('-----------------------------------------------------')
