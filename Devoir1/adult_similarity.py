# Xuanchen Liu - Matricule : 20173286
# Van Nam Vu - Matricule : 20170148
# -------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import statistics

df = pd.read_csv("adult.csv")  # Importer les donnees de ADULT
dfm = df.loc[:, df.columns != 'income']  # Enlever la colonne 'income'
X = df[dfm.columns]  # Diviser les donnees
y = df['income']

# Remplacer '<=50' et '>50', qui sont des valeurs dans la colonne 'income', par 0 et 1 respectivement
update = []
for index in y.keys():
    value = y.get(index)
    if value == '<=50K':
        update.append(0)
    else:
        update.append(1)
y.update(pd.Series(update))
y = y.astype('int')

# Diviser les donnees en donnees pour l'entrainement, test et validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
train_dim = 100  # Specifier la taille de train utilise
# Specifier respectivement au train la taille de test
test_dim = math.floor((train_dim / len(X_train)) * len(y_test))
X_train_reduced, y_train_reduced = X_train.head(train_dim), y_train.head(
    train_dim)  # Reduire la taille de train et test
X_test_reduced, y_test_reduced = X_test.head(test_dim), y_test.head(test_dim)

numerics = ['int16', 'int32', 'int64', 'float16',
            'float32', 'float64']  # Type de donnees a distinguer
# Distinguer les donnees continues
num_list = list(dfm.select_dtypes(include=numerics).columns)

# Trouver les valeurs maximal et minimal de chaque colonne continue
df_max = {}  # Stocker les valeurs maximal de chaque colonne
df_min = {}  # Stocker les valeurs minimal de chaque colonne
for var in num_list:
    df_max[var] = df[var].max()
    df_min[var] = df[var].min()


# Calculer la similarite entre deux groupe d'une meme column
# Parametre: feature: Attribut(colonne) a considerer / target: L'attribut(colonne) selon lequel le ratio est calculer
# Return value: Dictionnaire contenant la similarite entre tous les pairs de groupe de la colonne considere
def get_sim(feature, target):
    feature_count = df.groupby([feature, target])[feature].count()
    feature_number = df.groupby([feature])[feature].count()
    # Stocker les ratios des groupes {(String)Nom du groupe: (float)ratio du groupe}
    ratio_dictOrg = {}
    # Calcul du ratio formule: #personne avec <=50K salaire dans la colonne/ #personne total de la colonne
    for group in feature_number.keys():
        if group == '?':  # Ignorer les donnees manquantes
            continue
        else:
            ratio = (feature_count.get((group, 0))) / feature_number.get(group)
            ratio_dictOrg[group] = ratio

    ratio_dictMod = ratio_dictOrg  # Copy du ratio_dictOrg, utiliser dans le loop suivant
    # Stocker la similarite entre deux groupes {(frozenset)(Nom groupe1, nom groupe2):(float)similarite}
    sim_dict = {}
    for group1 in feature_number.keys():
        # Similarite avec tout les autre groupe est 0.5 si la valeur est manquante
        if group1 == '?':
            for group2 in feature_number.keys():
                sim_dict[frozenset((group1, group2))] = 0.5
        else:
            ratio1 = ratio_dictMod.get(group1)  # Ratio du groupe 1
            for group2 in ratio_dictMod.keys():
                if group2 == group1:  # Similarite 1 si meme groupe
                    sim_dict[frozenset((group1, group2))] = 1
                else:  # Similarite = 1 - diff(ratio groupe1, ratio groupe2)
                    # La similarite entre groupe1 et groupe2
                    sim_dict[frozenset((group1, group2))] = 1 - \
                        np.absolute(ratio1 - ratio_dictMod.get(group2))
            ratio_dictMod.pop(group1)
    return sim_dict

# Calculer la similarite de tout les columns avec valeur non-numerique
# Parametre: (Dataframe) donnees consideree
# Return: Dictionnaire contenant la similarite entre tous les pairs de groupe de toutes les colonnes dans dataframe


def cat_Sim(data):
    # Stocker les similarite {'Attribut':{(frozenset)(Groupe1, Groupe2):(float)Similarite}}
    sim_dict = {}
    for feature in data.columns:
        feature_sim = get_sim(feature, 'income')
        sim_dict[feature] = feature_sim
    return sim_dict

# Calculer les poids associe a chaque attribut(colonne)
#Parametre: (dictionnaire)
# Return: Dictionnaire contenant le poids associe a des attributs(colonnes)


def calculateWeight(lib):
    # Stocker la moyenne de similarite {'Attribut':(int)Moyenne de similarite}
    dict_avg = {}
    for feature in lib.keys():
        a = 0  # Valeur total de similarite
        for group in lib.get(feature).keys():
            a += lib.get(feature).get(group)
        dict_avg[feature] = a / len(lib.get(feature))
    val = []  # Stocker les moyennes en array
    for avg in dict_avg.values():
        val.append(avg)
    # Calculer la moyenne des moyennes de similarite
    mean = statistics.mean(val)
    stdev = statistics.stdev(val)  # Calculer l'ecart-type
    weight = {}  # Stocker les poids {'Attribut':(float)poids}
    for name in dict_avg.keys():
        value = dict_avg.get(name)  # La moyenne associe au attribut specifie
        ecart = np.absolute(mean - value)
        # Associe des poids aux attributs(colonnes)
        if value < mean and ecart > stdev:
            weight[name] = 0.11
        elif value < mean and ecart < stdev:
            weight[name] = 0.05
        elif value > mean and ecart < stdev:
            weight[name] = 0.02
        elif value > mean and ecart > stdev:
            weight[name] = 0.01
    return weight


# Distinguer les attributs discretes
df_cat = dfm.select_dtypes(exclude=numerics)
cat_list = list(df_cat.columns)  # List des noms des attributs discretes
sim_lib = cat_Sim(df_cat)  # Calculer les similarite
# Calculer les poids des attributs discretes
cat_weight = calculateWeight(sim_lib)

# Specifier des poids aux attributs continues
num_weight = {'age': 0.05, 'fnlwgt': 0.01, 'educational-num': 0.11,
              'capital-gain': 0.09, 'capital-loss': 0.09, 'hours-per-week': 0.24}

# Calculer la similarite entre deux personnes
# Parametre: (series)personne1, (series)personne2
# Return: (float)Similarite entre les deux personnes


def calculate_sim(x1, x2):
    cat_dict = {}  # Stocker les similarite des attributs discretes {'Attribut':similarite}
    # Ajouter les similarite dans cat_dict
    for feature in cat_list:
        # Chercher selon les groupes de l'attribut
        cat_dict[feature] = frozenset((x1.get(feature), x2.get(feature)))

    cat_sim = 0  # Similarite total des attributs discretes
    for dt in cat_dict.keys():
        dt_sim = sim_lib.get(dt).get(cat_dict.get(dt))
        # Ajouter les poids correspondants au attribut
        cat_sim += dt_sim*cat_weight.get(dt)

    num_sim = 0  # Similarite total des attributs continues
    for var in num_list:
        diff = np.absolute((x1.get(var) - x2.get(var)) /
                           (df_max.get(var) - df_min.get(var)))
        num_sim += (1 - diff)*num_weight.get(var)

    final_sim = num_sim + cat_sim  # Similatite total entre ces duex personnes
    return final_sim

# Matrice de disimilarite carre
# Parametre: (dataframe)donnees a considere
# Return: (array of array)Matrice de disimilarite


def dis_matrix(data):
    D = np.zeros((len(data), len(data)))  # Initialiser la matrice
    for i in range(len(data)):  # Ajouter les similarte dans la matrice
        for j in range(i, len(data)):
            if(i == j):
                D[i, j] = 0
            else:
                D[i, j] = 1 - calculate_sim(data.iloc[i], data.iloc[j])
                D[j, i] = D[i, j]
    return D

# Matrice de disimilarite non-carre
# Parametre: (dataframe,dataframe)donnees a considere
# Return: (array of array)Matrice de disimilarite


def dis_matrix_test(test, train):
    D = np.zeros((len(test), len(train)))  # Initialiser la matrice
    for i in range(len(test)):  # Ajouter les similarte dans la matrice
        for j in range(len(train)):
            D[i, j] = 1 - calculate_sim(test.iloc[i], train.iloc[j])
    return D

# Retourne les donnnes a utiliser dans les algo


def adult_import_data():
    train_matrix = dis_matrix(X_train_reduced)
    train_test_matrix = dis_matrix_test(X_test_reduced, X_train_reduced)
    return X_train_reduced, y_train_reduced, X_test_reduced, y_test_reduced, train_matrix, train_test_matrix
