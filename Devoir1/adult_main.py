# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Travailler sur dataset : Adults
# Fichier main pour generer les algorithmes
# et afficher les resultats pour evaluer la performance

import numpy as np
from adult_similarity import adult_import_data

from knn import knn_processing
from kmedoids_adult import kmedoids_processing
from isomap import isomap_processing
from pcoa import pcoa_processing
from partitionBinaire_adult import PB_processing


# fonction pour calculer l'exactitude de l'algorithme
def accuracy_mnist(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    succes = np.sum(a == b)
    return round(succes/len(y_predict), 5)


print('###------ Dataset ADULT ------####')
print("---importing data.....")
adultX_train, adultY_train, adultX_test, adultY_test, adultD_train, adultD_test = adult_import_data()
print('# Data train = ' + str(len(adultX_train)))
print('# Data test = ' + str(len(adultX_test)))
print(np.shape(adultD_train))
print(np.shape(adultD_test))
print("- imported ")
print('\n')

print('---------- 1) Algorithme KNN ----------')
print('... Processing ....')

knn_adult = knn_processing(adultD_train, adultD_test,
                           adultY_train, adultY_test)


print('----')
print('\n')

print('---------- 2) Algorithme K-medoids ----------')
print('... Processing ....')
kmedoids_adult = 2
print('Execute avec ' + str(kmedoids_adult)+'-medoids')
kmedoids_adult_predict = kmedoids_processing(
    adultX_train, adultY_train, adultD_train, adultD_test, kmedoids_adult)

print(' -> done processing')
print('-- Etape evaluation : ')

score_kmedoids = accuracy_mnist(kmedoids_adult_predict, adultY_test)
print('Accuracy = ' + str(score_kmedoids))
print('----')
print('\n')

print('---------- 3) Algorithme Isomap + application KNN ----------')
print('... Processing isomap ....')
n_comp_isomap = 6
print('Reduit dimentionnel à ' + str(n_comp_isomap))

isomap_adult = isomap_processing(
    adultD_train, adultD_test, adultY_train, adultY_test, n_comp_isomap)

print('----')
print('\n')

print('---------- 4) Algorithme PCoA + application KNN ----------')
print('... Processing isomap ....')

n_comp_pcoa = 6
print('Reduit dimentionnel à ' + str(n_comp_pcoa))

pcoa_adult = pcoa_processing(
    adultD_train, adultD_test, adultY_train, adultY_test, n_comp_pcoa)

print('----')
print('\n')

print('---------- 5) Algorithme Partition Binaire ----------')
print('... Processing ....')
cluster_adult = 2
print('Execute avec ' + str(cluster_adult)+' partitions')
pb_adult_predict = PB_processing(
    adultD_train, adultD_test, adultY_train, cluster_adult)

# pour avour array de prediction
score_partitionBinaire = accuracy_mnist(pb_adult_predict, adultY_test)
print('Accuracy = ' + str(score_partitionBinaire))
print('----')
print('\n')
print('-----------------------------------------------------')
