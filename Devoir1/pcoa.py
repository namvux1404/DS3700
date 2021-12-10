# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Implementation for PCoA
import numpy as np
# ce n'est pas PCoA mais on peut l'utiliser pour que le résultat soit le même
from sklearn.decomposition import KernelPCA
from sklearn import neighbors


def accuracy(y_predict, y_trueValue):  # Fonction pour calculer l'exactude
    a = np.array(y_predict)
    b = np.array(y_trueValue)

    succes = np.sum(a == b)

    return round(succes/len(y_predict), 5)


def pcoa_processing(D_train, D_test, y_train, y_test, n_comp):  # Fonction principale
    pcoa = KernelPCA(n_components=n_comp, kernel='precomputed')
    # -.5*D**2 est crucial!!!
    pcoa_train = pcoa.fit_transform(-.5*D_train**2)
    pcoa_test = pcoa.transform(-.5*D_test**2)  # -.5*D**2 est crucial!!!

    print(' -> done processing')
    print('-- Classification avec KNN --')
    score_train = []
    for k in range(1, 20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(pcoa_train, y_train)

        knn_mnist = knn.predict(pcoa_test)

        score = accuracy(knn_mnist, y_test)
        score_train.append(score)
        print('Accuracy = ' + str(score) + ' pour k = ' + str(k))
        print('-*-')

    print('Accuracy max = ' + str(np.max(score_train)))
    print('avec k = ' + str(np.argmax(score_train) + 1))
