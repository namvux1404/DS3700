# algorithm k-medoids.py

import pyclustering
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
from numpy.random import choice
from numpy.random import seed
import matplotlib.pyplot as plt
#import mnist_similarity
from mnist_similarity import mnist_dissimilarity
from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
#import preprocessing_mnist
#from preprocessing_mnist import import_data

# X = []  # data to be imported
# D = []  # table of dissimilarity
print('###------ Algorithme K-medoids ------####')
# D = mat_dissimilarity()  # recuperer la matrice de dissimilarite de tous les points
print("---importing data.....")
x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
print('# Data train = ' + str(len(x_train)))
print('# Data test = ' + str(len(x_test)))
print(np.shape(D_train))
print("- imported ")


def init_medoids(X, k):
    set_medoids = []
    seed(1)
    samples = choice(len(X), size=k, replace=False)

    for i in samples:
        set_medoids.append(i)

    return set_medoids


print("Step execute K-medoids")
initial_medoids = init_medoids(x_train, 10)
# for i in range(len(initial_medoids)):
#    print(y_train[initial_medoids[i]])

initial_medoids1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ici, nous donnons directement notre matrice de distance
# il faut spécifier data_type='distance_matrix'
kmedoids_instance = kmedoids(
    D_train, initial_medoids, data_type='distance_matrix')
print("-- processing....")
kmedoids_instance.process()

print(" - done")
clusters = kmedoids_instance.get_clusters()  # groupe pour chaque elements
medoids = kmedoids_instance.get_medoids()

print('Cluster form ' + str(np.shape(clusters)))
#print("Cluster : " + str(clusters))
print("Medoids : " + str(medoids))
#print('Medoides: -- Classe -- vrai valeur ')
# for i in medoids:
#    print(str(i) + "  -- +  +  ------ " +
#          str(y_train[i]))  # + str(clusters[i]))


def associate_clustering(clusters):
    associate_clustering = []

    for group in clusters:
        list_class = []  # initialize array
        for j in range(len(group)):
            list_class.append(y_train[group[j]])

        #print(' list class ' + str(list_class))
        most_frequent_class = np.bincount(list_class).argmax()
        associate_clustering.append(most_frequent_class)

    return associate_clustering


class_medoids = associate_clustering(clusters)
print('Class medoids : ' + str(class_medoids))


def accuracy(x_train, x_test, y_test, medoids, class_medoids):

    y_predict = []
    y_trueValue = []
    for i in range(len(x_test)):
        array = []
        for j in range(len(medoids)):
            # calcule dissimilarite avec les centroids
            array.append(mnist_dissimilarity(x_test[i], x_train[medoids[j]]))

        IndexminValue = np.argmin(array)  # retrieve index for class medoids
        y_predict.append(class_medoids[IndexminValue])
        y_trueValue.append(y_test[i])

    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    print(succes)
    return round(succes/len(x_test), 5)


print('Accuracy = ' + str(accuracy(x_train, x_test, y_test,
                                   medoids, class_medoids)))
