# Implementation for Isomap
# edited
import numpy as np
from sklearn.manifold import Isomap
from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
from sklearn import neighbors

print('###------ Algorithme isomap ------####')
print("---importing data.....")
x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
print('# Data train = ' + str(len(x_train)))
print('# Data test = ' + str(len(x_test)))
print(np.shape(D_train))
print(np.shape(D_test))
print("- imported ")

print('... Processing ....')
isomap = Isomap(n_components=8, n_neighbors=5, metric='precomputed')
isomap_mnist = isomap.fit_transform(D_train)

isomap_mnist_test = isomap.transform(D_test)

# print(isomap_mnist)
print(np.shape(isomap_mnist))


def accuracy(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    print(succes)
    return round(succes/len(y_predict), 5)


print('-- Classification avec KNN --')
score_train = []
for k in range(1, 20):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(isomap_mnist, y_train)
#print(" - done")
    print(" - Prediction and accuracy pour " + str(k))

# avec matrice de dissi avec l'ensemble test -> reduit dimesion
# -> knn.predict(isomao_mnist_test) format 500x2, pas isomao_mnist
    knn_mnist = knn.predict(isomap_mnist_test)
# print(knn_mnist)
# print(len(knn_mnist))
    score = accuracy(knn_mnist, y_test)
    score_train.append(score)
    print('Accuracy = ' + str(score))
    print('----')

print(" - done")
print(np.max(score_train))
print(np.argmax(score_train))
