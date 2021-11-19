# mnist_similarity_function.py

# import preprocessing_mnist
from preprocessing_mnist import import_data
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = import_data()


# print("----- Example Image 1 ------")
# print(y_train[0])
# Conversion de notre vecteur d'une dimension en 2 dimensions
# matrix = np.reshape(x_train[0], (28, 28))
# print(matrix)
# plt.imshow(matrix, cmap='gray')
# Affiche un 5, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()

# print("----- Example Image 2 ------")
# print(y_train[2])
# Conversion de notre vecteur d'une dimension en 2 dimensions
# matrix = np.reshape(x_train[2], (28, 28))

# print(matrix)
# plt.imshow(matrix, cmap='gray')

# Affiche un 5, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()


def print_image_original(X):  # Function for print image ori
    matrix = np.reshape(X, (28, 28))
    print(matrix)


# print("---- en translation ----")


# translation image a gauche et en haut --> enleve colonne pixel a gauche --> matrix 28x26
def image_translation_left(X):
    # mat_pixel0 = np.zeros((28, 1))
    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 0, axis=1)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 0, axis=0)  # delete 1 pixel en haut
    # mat = np.delete(mat, 0, axis=1)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 0, axis=1)  # delete 1 pixel a la fin
    # mat = np.append(mat, mat_pixel0, axis=1)        # ajoute 1 pixel au debut
    # print("----- Example Image 1 translation left ------")
    # print(np.shape(mat))
    # print(mat)
    return mat


# translation image a droite --> enleve colonne pixel a droite --> matrix 28x27
def image_translation_right(X):
    # mat_pixel0 = np.zeros((28, 1))
    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 27, axis=1)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 27, axis=0)  # delete 1 pixel en bas
    # mat = np.delete(mat, 26, axis=1)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 25, axis=1)  # delete 1 pixel a la fin
    # mat = np.append(mat, mat_pixel0, axis=1)        # ajoute 1 pixel au debut
    # print("----- Example Image 2 translation right ------")
    # print(np.shape(mat))
    # print(mat)
    return mat


# translation image en haut --> enleve ligne pixel en haut --> matrix 27x28
def image_translation_up(X):
    # mat_pixel0 = np.zeros((28, 1))
    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 0, axis=0)  # delete 1 pixel en haut
    # mat = np.delete(mat, 0, axis=0)  # delete 2 pixel en haut
    # mat = np.delete(mat, 0, axis=0)  # delete 1 pixel a la fin
    # mat = np.append(mat, mat_pixel0, axis=1)        # ajoute 1 pixel au debut
    # print("----- Example Image 1 translation up ------")
    # print(np.shape(mat))
    # print(mat)
    return mat


# translation image en bas --> enleve ligne pixel en bas --> matrix 27x28
def image_translation_bottom(X):
    # mat_pixel0 = np.zeros((28, 1))
    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 27, axis=0)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 26, axis=0)  # delete 1 pixel a la fin
    # mat = np.delete(mat, 25, axis=0)  # delete 1 pixel a la fin
    # mat = np.append(mat, mat_pixel0, axis=1)        # ajoute 1 pixel au debut
    # print("----- Example Image 2 translation bottom ------")
    # print(np.shape(mat))
    # print(mat)
    return mat


# image_modified_left = image_translation_left(x_train[0])
# print("----- Example Image 1 translation left ------")
# print(image_modified_left)
# plt.imshow(image_modified_left, cmap='gray')

# Affiche un 5 avec translation, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()

# image_modified_right = image_translation_right(x_train[2])
# print("----- Example Image 2 translation right ------")
# print(image_modified_right)
# plt.imshow(image_modified_right, cmap='gray')

# Affiche un 5 avec translation, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()

# image_modified_up = image_translation_up(x_train[0])
# print("----- Example Image 1 translation up ------")
# print(image_modified_up)
# plt.imshow(image_modified_right, cmap='gray')

# Affiche un 5 avec translation, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()

# image_modified_bottom = image_translation_bottom(x_train[2])
# print("----- Example Image 2 translation bottom ------")
# print(image_modified_bottom)
# plt.imshow(image_modified_right, cmap='gray')

# Affiche un 5 avec translation, tout comme nous avions vu comme premier "label" du jeu de données
# plt.show()

# print("Matrix difference")
# mat_diff_left_right = np.absolute(image_modified_left - image_modified_right)
# mat_diff_up_down = np.absolute(image_modified_up - image_modified_bottom)
# print(mat_diff_left_right)
# print(mat_diff_up_down)
# matrix_diff = []
# matrix_diff = np.append(matrix_diff, np.mean(mat_diff_left_right))
# matrix_diff = np.append(matrix_diff, np.mean(mat_diff_up_down))
# print(matrix_diff)
# print("valeur dissimilarite Im1 et Im3 = " +str(np.mean(matrix_diff)))


def mnist_dissimilarity(X, Y):
    # print("----- Example Image 1 ------")
    # print_image_original(X)
    # print("----- Example Image 2 ------")
    # print_image_original(Y)

    # print("---- en translation ----")
    # matrix image
    image_modified_left_X = image_translation_left(X)
    image_modified_right_Y = image_translation_right(Y)

    image_modified_right_X = image_translation_right(X)
    image_modified_left_Y = image_translation_left(Y)

    image_modified_up_X = image_translation_up(X)
    image_modified_bottom_Y = image_translation_bottom(Y)

    image_modified_up_Y = image_translation_up(Y)
    image_modified_bottom_X = image_translation_bottom(X)

    mat_diff_leftX_rightY = np.absolute(
        image_modified_left_X - image_modified_right_Y)
    mat_diff_leftY_rightX = np.absolute(
        image_modified_left_Y - image_modified_right_X)
    mat_diff_upX_downY = np.absolute(
        image_modified_up_X - image_modified_bottom_Y)
    mat_diff_upY_downX = np.absolute(
        image_modified_up_Y - image_modified_bottom_X)
    # print(mat_diff_left_right)
    # print(mat_diff_up_down)
    matrix_diff = []
    matrix_diff.append(round(
        np.mean(mat_diff_leftX_rightY), 2))
    matrix_diff.append(round(
        np.mean(mat_diff_leftY_rightX), 2))
    matrix_diff.append(round(
        np.mean(mat_diff_upX_downY), 2))
    matrix_diff.append(round(
        np.mean(mat_diff_upY_downX), 2))
    # print(matrix_diff)
    return round(np.mean(matrix_diff), 3)


def mat_dissimilarity(X):  # matrice des dissimilarite entre les images
    D = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            if (i == j):
                D[i, j] = 0  # meme image
            else:
                D[i, j] = mnist_dissimilarity(X[i], X[j])
                D[j, i] = D[i, j]  # criteria matrix symetrix

    return D


# matrice des dissimilarite entre les images
def mat_dissimilarity_test(x_test, x_train):

    D_test = np.zeros((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            # if (i == j):
            #    D[i, j] = 0  # meme image
            # else:
            diff = mnist_dissimilarity(x_test[i], x_train[j])
            #print('diff = ' + str(diff))
            D_test[i, j] = diff
            #    D[j, i] = D[i, j]  # criteria matrix symetrix

    return D_test


def mnist_import_Xtrain_Ytrain_matDiss():
    D_train = mat_dissimilarity(x_train)
    D_test = mat_dissimilarity_test(x_test, x_train)

    return x_train, y_train, x_test, y_test, D_train, D_test


#D_test = mat_dissimilarity_test(x_test, x_train)
# print(D_test)
# print_image_original(x_test[0])
# print_image_original(x_train[1])
#print(mnist_dissimilarity(x_test[0], x_train[1]))


# D = mat_dissimilarity(x_train)
# for i in range(5):
# print(D[i])

# print(mnist_dissimilarity(x_train[0], x_train[2]))
