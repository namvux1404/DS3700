import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import statistics

df = pd.read_csv("adult.csv")
dfm = df.loc[:,df.columns!='income'].head(10)
X = df[dfm.columns]
y = df['income']
update = []
for index in y.keys(): 
    value = y.get(index)
    if value == '<=50K':
        update.append(0)
    else:
        update.append(1)
y.update(pd.Series(update))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train_dim = 100
test_dim = math.floor((train_dim / len(X_train)) * len(y_test))
X_train_reduced, y_train_reduced = X_train.head(train_dim), y_train.head(train_dim)
X_test_reduced, y_test_reduced = X_test.head(test_dim), y_test.head(test_dim)

df_max = {}
df_min = {}
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
num_list = list(dfm.select_dtypes(include=numerics).columns)
for var in num_list:
    df_max[var] = df[var].max()
    df_min[var] = df[var].min()



#Calculer la similarite entre deux groupe d'une meme column
def get_sim(feature, target):
    feature_count = df.groupby([feature, target])[feature].count()
    feature_number = df.groupby([feature])[feature].count()
    ratio_dictOrg = {}
    for group in feature_number.keys():
        if group == '?':
            continue
        else:
            ratio = (feature_count.get((group, 0))) / feature_number.get(group)
            ratio_dictOrg[group] = ratio
        
    ratio_dictMod = ratio_dictOrg
    sim_dict = {}
    for group1 in feature_number.keys():
        if group1 == '?':
            for group2 in feature_number.keys():
                sim_dict[frozenset((group1,group2))] = 0.5
        else:
            ratio1 = ratio_dictMod.get(group1)
            for group2 in ratio_dictMod.keys():
                if group2 == group1:
                    sim_dict[frozenset((group1,group2))] = 1
                else:
                    sim_dict[frozenset((group1,group2))] = 1 - np.absolute(ratio1 - ratio_dictMod.get(group2)) #La similarite entre groupe1 et groupe2
            ratio_dictMod.pop(group1)
    return sim_dict

#Calculer la similarite de tout les columns avec valeur non-numerique
def cat_Sim(data):
    sim_dict = {}
    for feature in data.columns:
        feature_sim = get_sim(feature,'income')
        sim_dict[feature] = feature_sim
    return sim_dict


def calculateWeight(lib):
    dict_avg = {}
    for feature in lib.keys():
        a = 0
        for group in lib.get(feature).keys():
            a += lib.get(feature).get(group)
        dict_avg[feature] = a / len(lib.get(feature))
    val = []
    for avg in dict_avg.values():
        val.append(avg)
    mean = statistics.mean(val)
    stdev = statistics.stdev(val)
    weight = {}
    for name in dict_avg.keys():
        value = dict_avg.get(name)
        ecart = np.absolute(mean - value)
        max = 8/14
        if value < mean and ecart > stdev:
            weight[name] = 0.11
        elif value < mean and ecart < stdev:
            weight[name] = 0.05
        elif value > mean and ecart < stdev:
            weight[name] = 0.02
        elif value > mean and ecart > stdev:
            weight[name] = 0.01
    return weight


df_cat = dfm.select_dtypes(exclude=numerics) 
cat_list = list(df_cat.columns)   #Table avec les columns non-numeriques
sim_lib = cat_Sim(df_cat)
cat_weight = calculateWeight(sim_lib)
num_weight = {'age':0.05,'fnlwgt':0.01,'educational-num':0.11,'capital-gain':0.09,'capital-loss':0.09,'hours-per-week':0.24}

#Calculer la similarite entre deux elements
def calculate_sim(x1,x2):
    cat_dict = {}
    for feature in cat_list:
        cat_dict[feature] = frozenset((x1.get(feature),x2.get(feature)))
    cat_sim = 0
    for dt in cat_dict.keys():
        dt_sim = sim_lib.get(dt).get(cat_dict.get(dt))
        cat_sim += dt_sim*cat_weight.get(dt)
    num_sim = 0
    for var in num_list:
        diff = np.absolute((x1.get(var) - x2.get(var)) / (df_max.get(var) - df_min.get(var)))
        num_sim += (1 - diff)*num_weight.get(var)
    final_sim = num_sim + cat_sim
    return final_sim
    
    
def dis_matrix(data):
    D = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(i,len(data)):
            if(i == j):
                D[i,j] = 0
            else:
                D[i,j] = 1 - calculate_sim(data.iloc[i],data.iloc[j])
                D[j,i] = D[i,j]
    return D


def dis_matrix_test(test,train):
    D = np.zeros((len(test),len(train)))
    for i in range(len(test)):
        for j in range(len(train)):
            D[i,j] = 1 - calculate_sim(test.iloc[i],train.iloc[j])
    return D
            
def adult_import_data():
    train_matrix = dis_matrix(X_train_reduced)
    train_test_matrix = dis_matrix_test(X_test_reduced,X_train_reduced)
    return X_train_reduced, y_train_reduced, X_test_reduced, y_test_reduced,train_matrix,train_test_matrix   
     
