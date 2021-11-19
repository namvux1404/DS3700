# Implementation for PCoA
import numpy as np
from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
# ce n'est pas PCoA mais on peut l'utiliser pour que le résultat soit le même
from sklearn.decomposition import KernelPCA
from sklearn import neighbors


print('###------ Algorithme PCoA ------####')
print("---importing data.....")
x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
print('# Data train = ' + str(len(x_train)))
print('# Data test = ' + str(len(x_test)))
print(np.shape(D_train))
print("- imported ")

print('... Processing ....')
pcoa = KernelPCA(n_components=8, kernel='precomputed')
# -.5*D**2 est crucial!!!
pcoa_mnist_train = pcoa.fit_transform(-.5*D_train**2)
pcoa_mnist_test = pcoa.transform(-.5*D_test**2)  # -.5*D**2 est crucial!!!

# print(pcoa_mnist)
# print(np.shape(pcoa_mnist))


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
for k in range(5, 20):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(pcoa_mnist_train, y_train)
#print(" - done")
    print(" - Prediction and accuracy pour " + str(k))

# avec matrice de dissi avec l'ensemble test -> reduit dimesion
# -> knn.predict(isomao_mnist_test) format 500x2, pas isomao_mnist
    knn_mnist = knn.predict(pcoa_mnist_test)
# print(knn_mnist)
# print(len(knn_mnist))
    score = accuracy(knn_mnist, y_test)
    score_train.append(score)
    print('Accuracy = ' + str(score))
    print('----')

print(" - done")
print(np.max(score_train))
print(np.argmax(score_train))
