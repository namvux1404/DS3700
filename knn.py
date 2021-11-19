# Implement KNN
import numpy as np
#from mnist_similarity import mnist_dissimilarity
from mnist_similarity import mnist_import_Xtrain_Ytrain_matDiss
from sklearn.neighbors import KNeighborsClassifier  # k-plus proches voisins


def accuracy(y_predict, y_trueValue):
    a = np.array(y_predict)
    b = np.array(y_trueValue)
    # print(y_predict)
    # print(y_test)
    succes = np.sum(a == b)
    print(succes)
    return round(succes/len(y_predict), 5)


print('\n')
print('###------ Algorithme KNN ------####')
print("---importing data.....")
x_train, y_train, x_test, y_test, D_train, D_test = mnist_import_Xtrain_Ytrain_matDiss()
print('# Data train = ' + str(len(x_train)))
print('# Data test = ' + str(len(x_test)))
print(np.shape(D_train))
print(np.shape(D_test))
print("- imported ")

# produire autre array D avec x_test et x_train pour predict

print('... Processing ....')
score_train = []
for k in range(1, 30):
    knn = KNeighborsClassifier(
        n_neighbors=k, metric='precomputed', algorithm='brute')
    knn.fit(D_train, y_train)
    #print(" - done processing")
    print(" - Prediction and accuracy pour " + str(k))

    knn_mnist = knn.predict(D_test)
    # print(knn_mnist)
    # print(len(knn_mnist))
    score = accuracy(knn_mnist, y_test)
    score_train.append(score)
    print('Accuracy = ' + str(score))
    print('----')

print(" - done")
print(np.max(score_train))
print(np.argmax(score_train))
