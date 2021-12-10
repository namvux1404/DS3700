# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# algorithm k-medoids.py

import pyclustering
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
from numpy.random import choice
from numpy.random import seed
import matplotlib.pyplot as plt


# pour initialiser les medoids
# @param
# X : ensemble train
# k : nombre de medoids

# pour initialiser les centroids
def init_medoids(X, k):
    set_medoids = []
    seed(1)
    samples = choice(len(X), size=k, replace=False)

    for i in samples:
        set_medoids.append(i)

    return set_medoids


# Fonction pour determiner les classe des clusters
def associate_clustering(y_train, array_clusters, n_clusters):
    class_clusters = []
    for i in range(n_clusters):
        searchval = i  # cluster i

        # les points de meme partition
        array_points = np.where(array_clusters == searchval)[0]
        list_class = []  # array les class de ce meme partition
        for j in range(len(array_points)):
            list_class.append(y_train[array_points[j]])

        #print(' list class ' + str(list_class))
        most_frequent_class = np.bincount(list_class).argmax()
        class_clusters.append(most_frequent_class)

    return class_clusters


# Fonction pour sortir array prédiction pour x_test
def y_prediction(array_clusters, class_clusters):
    y_predict = []  # array contient classification selon partition
    for i in range(len(array_clusters)):
        partition = array_clusters[i]  # partition predict de element i

        class_cluster_predict = class_clusters[partition]

        # ajout dans array y_predict pour valider
        y_predict.append(class_cluster_predict)
    return y_predict


def kmedoids_processing(x_train, y_train, D_train, D_test, k):  # fonction principale
    y_predict = []

    initial_medoids = init_medoids(x_train, k)

    kmedoids_instance = kmedoids(
        D_train, initial_medoids, data_type='distance_matrix')

    kmedoids_instance.process()

    kmedoids_train = kmedoids_instance.predict(D_train)

    kmedoids_test = kmedoids_instance.predict(D_test)

    class_medoids = associate_clustering(y_train, kmedoids_train, k)
    #print('Class medoids : ' + str(class_medoids))

    y_predict = y_prediction(kmedoids_test, class_medoids)
    return y_predict
