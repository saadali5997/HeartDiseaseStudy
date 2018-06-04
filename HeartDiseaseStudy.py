import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import itertools
from pprint import pprint

columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", sep=',', header=None,
                   names=columns)

df = df.replace('?', np.nan)
df = df.dropna()

# Convert categorical variables with more than two values into dummy variables
dummies = pd.get_dummies(df["cp"], prefix="cp")
df = df.join(dummies)
del df["cp"]
df = df.rename(columns={"cp_1.0": "cp_1", "cp_2.0": "cp_2", "cp_3.0": "cp_3", "cp_4.0": "cp_4"})

dummies = pd.get_dummies(df["restecg"], prefix="recg")
df = df.join(dummies)
del df["restecg"]
df = df.rename(columns={"recg_0.0": "recg_0","recg_1.0": "recg_1", "recg_2.0": "recg_2"})

dummies = pd.get_dummies(df["slope"], prefix="slope")
df = df.join(dummies)
del df["slope"]
df = df.rename(columns={"slope_1.0": "slope_1", "slope_2.0": "slope_2","slope_3.0": "slope_3"})

dummies = pd.get_dummies(df["thal"], prefix="thal")
df = df.join(dummies)
del df["thal"]
df = df.rename(columns={"thal_3.0": "thal_3","thal_6.0": "thal_6", "thal_7.0": "thal_7"})
result=df['num']
df=df.drop("num",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, result, 
                                                    train_size=0.75, 
                                                    random_state=150)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 20)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, Y_train)
    # record training set accuracys
    training_accuracy.append(clf.score(X_train, Y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, Y_test))

print(training_accuracy)
print(test_accuracy)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB(priors=None)
clf.fit(X_train, Y_train)

pred = clf.predict(X_train)
print (accuracy_score(pred, Y_train))

pred = clf.predict(X_test)
print (accuracy_score(pred, Y_test))
#print ("Testing Accuracy of the prediction:"+accuracy_score(pred, Y_test))

from sklearn.tree import DecisionTreeClassifier


for mdp in range(1,15):
    for mln in range(2,23):
        tree = DecisionTreeClassifier(max_depth=mdp,random_state=5,max_leaf_nodes=mln)
        tree.fit(X_train, Y_train)
        print(mdp)
        print(mln)
        print("Accuracy on training set: {:.3f}".format(tree.score(X_train, Y_train)))
        print("Accuracy on test set: {:.3f}".format(tree.score(X_test, Y_test)))

from sklearn.ensemble import RandomForestClassifier

lst=[]


for nest in range(1,15):
    for mdp in range(1,7):
        for mft in range(1,23):
            forest = RandomForestClassifier(n_estimators=nest, random_state=10, max_depth=mdp, max_features=mft)
            forest.fit(X_train, Y_train)
            max_accuracy={}
            max_accuracy['n_estimator']=nest
            max_accuracy['max_depth']=mdp
            max_accuracy['max_features']=mft
            max_accuracy['t_accuracy']=forest.score(X_train, Y_train)
            max_accuracy['test_accuracy']=forest.score(X_test, Y_test)
            lst.append(max_accuracy)
            del max_accuracy
        
newlist = sorted(lst, key=lambda k: k['test_accuracy']) 

for dictionary in newlist:
    print(dictionary)
