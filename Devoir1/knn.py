# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# Implement KNN

import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # k-plus proches voisins


def accuracy(y_predict, y_trueValue):  # Fonction pour calculer l'exactude
    a = np.array(y_predict)
    b = np.array(y_trueValue)

    succes = np.sum(a == b)

    return round(succes/len(y_predict), 5)


def knn_processing(D_train, D_test, y_train, y_test):
    score_train = []
    for k in range(1, 20):
        knn = KNeighborsClassifier(
            n_neighbors=k, metric='precomputed', algorithm='brute')
        knn.fit(D_train, y_train)

        knn_predict = knn.predict(D_test)

        score = accuracy(knn_predict, y_test)
        score_train.append(score)
        print('Accuracy = ' + str(score) + ' pour k = ' + str(k))
        print('----')
    print(' -> done processing')
    print('Accuracy max = ' + str(np.max(score_train)))
    print('avec k = ' + str(np.argmax(score_train) + 1))
