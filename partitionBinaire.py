# Implementation Partition binaire
# Regroupement hiérarchique (Paritionnement binaire)
from sklearn.cluster import AgglomerativeClustering
import numpy as np
#from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
#import scipy.cluster._hierarchy as shc
import matplotlib.pyplot as plt

# print('###------ Algorithme Partition Binaire ------####')
#print("---importing data.....")
#x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
# print('# Data train = ' + str(len(x_train)))
# print('# Data test = ' + str(len(x_test)))
# print(np.shape(D_train))
# print(np.shape(D_test))
#print("- imported ")

#print('... Processing ....')


def accuracy(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    print(succes)
    return round(succes/len(y_predict), 5)


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

    # print(agglo_mnist)  # --> besoin d'association a classe
    #print('size array agglo_mnist = ' + str(len(agglo_mnist)))

    # pour test
    agglo_mnist_test = agglomerative_clustering_predict(
        agglomerative_clustering, D_test)

    #print('size array agglo_mnist_test = ' + str(len(agglo_mnist_test)))
    # print(agglo_mnist_test)
    # appel une autre fois pour dissim de test et determine class
    # avec l'association de train --> compare avec y_test

    #print(" - done")
    # pour determiner les class de partition,array_partition sous forme np.array

    classification_partition = class_partition(y_train, agglo_mnist, clusters)
    print('Class partition : ' + str(classification_partition))
    # print(classification_partition)

    #print(" - Prediction and accuracy : ")

    y_predict = y_prediction(agglo_mnist_test, classification_partition)
    return y_predict
    # print(y_predict)

    #score = accuracy(y_predict, y_test)
    # print('Accuracy = ' + str(score))
    # print('----')
