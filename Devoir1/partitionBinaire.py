# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Implementation Partition binaire
# Regroupement hiérarchique (Paritionnement binaire)
from sklearn.cluster import AgglomerativeClustering
import numpy as np


# Fonction pour determiner les classe des partition
def class_partition(y_train, array_partition, n_clusters):
    class_partition = []
    for i in range(n_clusters):
        searchval = i  # partition i

        # les points de meme partition
        array_points = np.where(array_partition == searchval)[0]
        list_class = []  # array les class de ce meme partition
        for j in range(len(array_points)):
            list_class.append(y_train[array_points[j]])

        #print(' list class ' + str(list_class))
        most_frequent_class = np.bincount(list_class).argmax()
        class_partition.append(most_frequent_class)

    return class_partition


# Fonction pour sortir array prédiction pour x_test
def y_prediction(array_partition, classification):
    y_predict = []  # array contient classification selon partition
    for i in range(len(array_partition)):
        partition = array_partition[i]  # partition predict de element i
        # print(partition)
        class_parition_predict = classification[partition]
        # print(class_parition_predict)

        # ajout dans array y_predict pour valider
        y_predict.append(class_parition_predict)
    return y_predict


# Fonction principale
def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(
            agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)


def PB_processing(D_train, D_test, y_train, clusters):
    agglomerative_clustering = AgglomerativeClustering(
        n_clusters=clusters, affinity='precomputed', linkage='average')
    agglomerative_clustering.fit(D_train)

    # pour train
    agglo_mnist = agglomerative_clustering_predict(
        agglomerative_clustering, D_train)

    # pour test
    agglo_mnist_test = agglomerative_clustering_predict(
        agglomerative_clustering, D_test)

    classification_partition = class_partition(y_train, agglo_mnist, clusters)
    print('Class partition : ' + str(classification_partition))

    y_predict = y_prediction(agglo_mnist_test, classification_partition)
    return y_predict
