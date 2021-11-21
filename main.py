# Xuan - Matricule
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Travailler sur 2 datasets : MNIST et Adults
# Fichier main pour generer les algorithmes
# et afficher les resultats pour evaluer la performance

#from pyclustering.cluster.kmedoids import kmedoids
#from numpy.random import choice
#from numpy.random import seed

#from sklearn.cluster import AgglomerativeClustering
#

#import scipy.cluster._hierarchy as shc
#import matplotlib.pyplot as plt

# from sklearn.neighbors import KNeighborsClassifier  # k-plus proches voisins

#from sklearn.manifold import Isomap
#from sklearn import neighbors
import numpy as np
from mnist_similarity import mnist_import_data
from knn import knn_processing
from kmedoids import kmedoids_processing
# fonction pour calculer l'exactitude de l'algorithme


def accuracy_mnist(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    # print(succes)
    return round(succes/len(y_predict), 5)


print('###------ Dataset MNIST ------####')
print("---importing data.....")
mnistX_train, mnistY_train, mnistX_test, mnistY_test, mnistD_train, mnistD_test = mnist_import_data()
print('# Data train = ' + str(len(mnistX_train)))
print('# Data test = ' + str(len(mnistX_test)))
print(np.shape(mnistD_train))
print(np.shape(mnistD_test))
print("- imported ")
print('\n')

print('---------- 1) Algorithme KNN ----------')
print('... Processing ....')
kneighbors_mnist = 1
print('Execute avec ' + str(kneighbors_mnist)+'-NN')
knn_mnist = knn_processing(mnistD_train, mnistD_test,
                           mnistY_train, kneighbors_mnist)
print(' -> done processing')
print('-- Etape evaluation : ')
# print(knn_mnist)
score = accuracy_mnist(knn_mnist, mnistY_test)
print('Accuracy = ' + str(score))
print('----')

print('\n')
print('---------- 2) Algorithme K-medoids ----------')
print('... Processing ....')
kmedoids_mnist = 10
print('Execute avec ' + str(kmedoids_mnist)+'-medoids')
kmedoids_mnist_predict = kmedoids_processing(
    mnistD_train, mnistY_train, mnistD_train, mnistD_test, kmedoids_mnist)

print(' -> done processing')
print('-- Etape evaluation : ')
# print(knn_mnist)
score = accuracy_mnist(kmedoids_mnist_predict, mnistY_test)
print('Accuracy = ' + str(score))
print('----')
