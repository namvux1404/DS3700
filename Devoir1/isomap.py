# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Implementation for Isomap
import numpy as np
from sklearn.manifold import Isomap
from sklearn import neighbors


def accuracy(y_predict, y_trueValue):  # Fonction pour calculer l'exactude
    a = np.array(y_predict)
    b = np.array(y_trueValue)

    succes = np.sum(a == b)

    return round(succes/len(y_predict), 5)


def isomap_processing(D_train, D_test, y_train, y_test, n_comp):  # Fonction principale
    isomap = Isomap(n_components=n_comp, n_neighbors=5, metric='precomputed')
    isomap_train = isomap.fit_transform(D_train)

    isomap_test = isomap.transform(D_test)

    print(' -> done processing')

    print('-- Classification avec KNN --')
    score_train = []
    for k in range(1, 20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(isomap_train, y_train)

        knn_predict = knn.predict(isomap_test)

        score = accuracy(knn_predict, y_test)
        score_train.append(score)
        print('Accuracy = ' + str(score) + ' pour k = ' + str(k))
        print('-*-')

    print('Accuracy max = ' + str(np.max(score_train)))
    print('avec k = ' + str(np.argmax(score_train) + 1))
