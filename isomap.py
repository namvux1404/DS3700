# Implementation for Isomap
# edited
import numpy as np
from sklearn.manifold import Isomap
#from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
from sklearn import neighbors

# print('###------ Algorithme isomap ------####')
#print("---importing data.....")
#x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
# print('# Data train = ' + str(len(x_train)))
# print('# Data test = ' + str(len(x_test)))
# print(np.shape(D_train))
# print(np.shape(D_test))
#print("- imported ")


def accuracy(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    # print(succes)
    return round(succes/len(y_predict), 5)


#print('... Processing ....')
def isomap_processing(D_train, D_test, y_train, y_test, n_comp):
    isomap = Isomap(n_components=n_comp, n_neighbors=5, metric='precomputed')
    isomap_train = isomap.fit_transform(D_train)

    isomap_test = isomap.transform(D_test)

    # print(isomap_mnist)
    # print(np.shape(isomap_mnist))
    print(' -> done processing')

    print('-- Classification avec KNN --')
    score_train = []
    for k in range(1, 20):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(isomap_train, y_train)
        #print(" - done")
        #print(" - Prediction and accuracy pour k = " + str(k))

        # avec matrice de dissi avec l'ensemble test -> reduit dimesion
        # -> knn.predict(isomao_mnist_test) format 500x2, pas isomao_mnist
        knn_predict = knn.predict(isomap_test)
        # return knn_predict
        # print(knn_mnist)
        # print(len(knn_mnist))
        score = accuracy(knn_predict, y_test)
        score_train.append(score)
        print('Accuracy = ' + str(score) + ' pour k = ' + str(k))
        print('-*-')

        #print(" - done")
    print('Accuracy max = ' + str(np.max(score_train)))
    print('avec k = ' + str(np.argmax(score_train) + 1))
