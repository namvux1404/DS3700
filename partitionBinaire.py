# Implementation Partition binaire
# Regroupement hiérarchique (Paritionnement binaire)
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
#import scipy.cluster._hierarchy as shc
import matplotlib.pyplot as plt

print('###------ Algorithme Partition Binaire ------####')
print("---importing data.....")
x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
print('# Data train = ' + str(len(x_train)))
print('# Data test = ' + str(len(x_test)))
print(np.shape(D_train))
print(np.shape(D_test))
print("- imported ")

print('... Processing ....')


def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(
            agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)


agglomerative_clustering = AgglomerativeClustering(
    n_clusters=13, affinity='precomputed', linkage='average')
agglomerative_clustering.fit(D_train)


agglo_mnist = agglomerative_clustering_predict(
    agglomerative_clustering, D_train)
print(agglo_mnist)  # --> besoin d'association a classe

# appel une autre fois pour dissim de test et determine class
# avec l'association de train --> compare avec y_test

print(" - done")

print(" - Prediction and accuracy : ")


# pour determiner les class de partition,array_partition sous forme np.array
def class_partition(array_partition):
    class_partition = []
    for i in range(13):
        searchval = i
        array_points = np.where(array_partition == searchval)[0]
        list_class = []
        for j in range(len(array_points)):
            list_class.append(y_train[array_points[j]])

        #print(' list class ' + str(list_class))
        most_frequent_class = np.bincount(list_class).argmax()
        class_partition.append(most_frequent_class)

    return class_partition


classification_partition = class_partition(agglo_mnist)
print(classification_partition)


def accuracy(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    print(succes)
    return round(succes/len(y_predict), 5)


#score = accuracy(agglo_mnist, y_train)
#print('Accuracy = ' + str(score))
# print('----')
