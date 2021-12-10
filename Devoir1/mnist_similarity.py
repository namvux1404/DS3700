# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
# mnist_similarity_function.py

from preprocessing_mnist import import_data
import numpy as np
from scipy.spatial import distance

x_train, y_train, x_test, y_test = import_data()


def print_image_original(X):  # Function for print image ori
    matrix = np.reshape(X, (28, 28))
    print(matrix)

# -------- les fonctions de translation-------------


# translation image a gauche --> enleve 1ere colonne pixel--> matrix 28x27
def image_translation_left(X):
    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 0, axis=1)  # delete colonne pixel au debut

    return mat


# translation image a droite --> enleve derniere colonne pixel  --> matrix 28x27
def image_translation_right(X):

    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 27, axis=1)  # delete colonne pixel a la fin

    return mat


# translation image en haut --> enleve 1er ligne pixel--> matrix 27x28
def image_translation_up(X):

    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 0, axis=0)  # delete ligne pixel au début

    return mat


# translation image en bas --> enleve derniere ligne pixel --> matrix 27x28
def image_translation_bottom(X):

    mat = np.reshape(X, (28, 28))
    mat = np.delete(mat, 27, axis=0)  # delete 1 pixel a la fin

    return mat


# Fonction de calculer la dissimilarité entre image X et image Y
def mnist_dissimilarity(X, Y):

    image_modified_left_X = image_translation_left(
        X)  # Image X translate à gauche
    image_modified_right_Y = image_translation_right(
        Y)  # Image Y translate à droite

    image_modified_right_X = image_translation_right(
        X)  # Image X translate à droite
    image_modified_left_Y = image_translation_left(
        Y)  # Image Y translate à gauche

    image_modified_up_X = image_translation_up(X)  # Image X translate en haut
    image_modified_bottom_Y = image_translation_bottom(
        Y)  # Image Y translate en bas

    image_modified_up_Y = image_translation_up(Y)  # Image Y translate en haut
    image_modified_bottom_X = image_translation_bottom(
        X)  # Image X translate en bas

    # matrice de différence apres la translation
    mat_diff_leftX_rightY = np.absolute(
        image_modified_left_X - image_modified_right_Y)

    # matrice de différence apres la translation
    mat_diff_leftY_rightX = np.absolute(
        image_modified_left_Y - image_modified_right_X)

    # matrice de différence apres la translation
    mat_diff_upX_downY = np.absolute(
        image_modified_up_X - image_modified_bottom_Y)

    # matrice de différence apres la translation
    mat_diff_upY_downX = np.absolute(
        image_modified_up_Y - image_modified_bottom_X)

    matrix_diff = []

    # moyenne_1
    matrix_diff.append(round(
        np.mean(mat_diff_leftX_rightY), 2))

    # moyenne_2
    matrix_diff.append(round(
        np.mean(mat_diff_leftY_rightX), 2))

    # moyenne_3
    matrix_diff.append(round(
        np.mean(mat_diff_upX_downY), 2))

    # moyenne_4
    matrix_diff.append(round(
        np.mean(mat_diff_upY_downX), 2))

    return round(np.sum(matrix_diff), 3)


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


# matrice des dissimilarite entre les images train et test
def mat_dissimilarity_test(x_test, x_train):

    D_test = np.zeros((len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            diff = mnist_dissimilarity(x_test[i], x_train[j])
            D_test[i, j] = diff

    return D_test


def mat_euclidean(X, Y=None):  # matrice de la distance euclidienne des images
    Y = X if Y is None else Y
    D_euclidien = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            D_euclidien[i, j] = distance.euclidean(X[i], Y[j])

    return D_euclidien


def mnist_import_data():  # Fonction pour exporter les données
    D_train = mat_dissimilarity(x_train)
    D_test = mat_dissimilarity_test(x_test, x_train)
    DE_train = mat_euclidean(x_train)
    DE_test = mat_euclidean(x_test, x_train)

    return x_train, y_train, x_test, y_test, D_train, D_test, DE_train, DE_test
