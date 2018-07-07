
#We have another heart disease dataset with 76 featuers. Usually, more features mean more information. Probably, we will be able to achieve better classification accuracy. Do all features give better  prediction? Or, can we ignore some of the features to achieve better accuracy. How many features give us the best classification accuracy? Which model works best with the optimized set of features? Which hypermeters are best for each model we choose? These are some of the questions that intrigue us. You have to explore these questions and come up with reasonable answers. You will submit your notebook with all the working. Your grading will be done on the level of detail and best answers you achieve.

# The data is present at [UCI machine learning data set](http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data). Some detail about the data is present at [link](http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names).
 
# I will recommend to read Chapter 4 of 'Introduction to machine learning using Python' book by Andreas C Mueller. It notebook can be found at [link](https://github.com/amueller/introduction_to_ml_with_python)

# # BSEF14M518 - SAAD ALI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools
import types
from pprint import pprint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier


# ### Data Reading

pd.options.display.max_rows=300
pd.options.display.max_columns=76 

col_temp = ["col1"]
columns = ["id","ccf","age", "sex","painloc","painexer","relrest","pncadn","cp","trestbps","htn","chol","smoke","cigs","years",
           "fbs","dm","famhist","restecg","ekgmo","ekgday","ekgyr","dig","prop","nitr","pro","diuretic","proto",
           "thaldur","thaltime","met","thalach","thalrest","tpeakbps","tpeakbpd","dummy","trestbpd","exang","xhypo",
           "oldpeak","slope","rldv5","rldv5e","ca","restckm","exerckm","restef","restwm","exeref","exerwm", "thal",
           "thalsev","thalpul","earlobe","cmo","cday","cyr","num","lmt","ladprox","laddist","diag","cxmain","ramus",
           "om1","om2","rcaprox","rcadist","lvx1","lvx2","lvx3","lvx4","lvf","cathef","junk","name"]

tdf = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data", sep='\n',
                 header=None,names=col_temp,encoding = "ISO-8859-1")

df=pd.DataFrame(columns=columns)

#One Row of our Dataframe to be consists of 10 rows of data read from link
#This for loop will process one chunk (10 rows) of CSV into one row of our dataframe to be

for i in range(0,len(tdf)):
    flag=False
    #First
    temp=tdf.iloc[i]['col1']
    temp=str(temp)
    temp=temp.split(' ')
    if(len(temp) != 7):
            i+=10
            continue
    row=temp
    
    #SecondToNineth
    for j in range(1,9):
        temp=tdf.iloc[i+j]['col1']
        temp=str(temp)
        temp=temp.split(' ')
        if(len(temp) != 8):
            i+=10
            flag=True
            break
        else: 
            row+=temp

    if flag:
        continue
        
    #Tenth
    temp=tdf.iloc[i+9]['col1']
    temp=str(temp)
    temp=temp.split(' ')
    if(len(temp) != 5):
            i+=10
            continue
    row+=temp
    
    
    for k in range(0,len(row)-1):
        row[k]=float(row[k])

    df.loc[int((i+9)/10)]=row
    
# ### Pre-Processing of Data

#Deleting Columns Which are not used or are irrelevant (As described here: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
df1=df.copy(deep=True)
del df1['lvx1']
del df1['lvx2']
del df1['lvx3']
del df1['lvx4']
del df1['lvf']
del df1['cathef']
del df1['junk']
del df1['name']
del df1['thalsev']
del df1['thalpul']
del df1['earlobe']
del df1['restckm']
del df1['exerckm']
del df1['dummy']
del df1['ccf']

#Replacing all missing values (-9s) with NaNs
df1=df1.replace(-9,np.nan)

#Deleting all columns with values missing in 80% or more of the rows
df1 = df1.loc[:, df1.isnull().mean() < .8]

#Listing all Categorical Features
cat_indeces = {}
cat_indeces['cp']=df1.columns.get_loc('cp')
cat_indeces['fbs']=df1.columns.get_loc('fbs')
cat_indeces['famhist']=df1.columns.get_loc('famhist')
cat_indeces['restecg']=df1.columns.get_loc('restecg')
cat_indeces['dig']=df1.columns.get_loc('dig')
cat_indeces['prop']=df1.columns.get_loc('prop')
cat_indeces['nitr']=df1.columns.get_loc('nitr')
cat_indeces['pro']=df1.columns.get_loc('pro')
cat_indeces['diuretic']=df1.columns.get_loc('diuretic')
cat_indeces['proto']=df1.columns.get_loc('proto')
cat_indeces['exang']=df1.columns.get_loc('exang')
cat_indeces['xhypo']=df1.columns.get_loc('xhypo')
cat_indeces['thal']=df1.columns.get_loc('thal')
cat_indeces['slope']=df1.columns.get_loc('slope')

#Replacing all missing categorical values with most repeated value
for key,value in cat_indeces.items():
    max_val = df1[key].value_counts().idxmax()
    df1[key]=[max_val if np.isnan(i) else i for i in df1[key]]

cat_indeces=list(cat_indeces.keys())

#Replacing All Missing Numeric Values with Mean (Mean is giving the best results, tried all three) 
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
for i in list(df1):
    if i not in cat_indeces:
        df1[[i]]=imputer.fit_transform(df1[[i]])
del cat_indeces

#Converting Categorical Values
dummies = pd.get_dummies(df1["cp"], prefix="cp")
df1 = df1.join(dummies)
del df1["cp"]

dummies = pd.get_dummies(df1["restecg"], prefix="recg")
df1 = df1.join(dummies)
del df1["restecg"]

dummies = pd.get_dummies(df1["proto"], prefix="proto")
df1 = df1.join(dummies)
del df1["proto"]

dummies = pd.get_dummies(df1["thal"], prefix="thal")
df1 = df1.join(dummies)
del df1["thal"]

dummies = pd.get_dummies(df1["slope"], prefix="slope")
df1 = df1.join(dummies)
del df1["slope"]

'''The response variable (num) is categorical with 5 values'''

results=df1['num']
del df1['num']

'''Used this model to compare the results of replacing missing numerical values with 
mean, mode and most_frequent...Mean gave best accuracy'''

forest = RandomForestClassifier()
scores = cross_val_score(forest, df1, results, cv=10, scoring='accuracy')
print(scores.mean())
df1

# ### Feature Engineering (Polynomial Method)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

#Rescale Data
poly = PolynomialFeatures(degree=2).fit(df1)
X_poly = poly.transform(df1)

# ### Feature Selection
# 
# Now we will be performing feature selection. Three ways to do so are
# ###### 1- Univariant Selection
# ###### 2- Recursive Elimination
# ###### 3- PCA (Principal Component Analysis
# 
# I will use univariant selection for feature selection in this problem
# 
# ### Univariant Selection

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

'''This Block of code runs all the models on different parameters and different number of features iteratively.
At the end of it's execution, it returns a list of touples of which model performed best at certain number 
of features with what parameters.
It also returns the model which performed best without performing feature selection!
It will take some time (specially because of Random Forest, be patient while it executes.''' 

#NoOfFeatures in percentile form. (10,15,20,30,40,50,60,70,80,90,100)
Nof=[0.6,0.9,1.2,1.85,2.5,3.1,3.7,4.35,5.0,5.6,6.25]

#List of information of models which performed best at different number of features
withs = []
flag=True
for i in Nof:
    #Feature Selection
    select = SelectPercentile(percentile=i)
    select.fit(X_poly, results)
    X_selected = select.transform(X_poly)

    #Scaling
    scaler = MinMaxScaler()
    X_selected_s = scaler.fit_transform(X_selected)
    X_poly_scale= scaler.fit_transform(X_poly)
    touple={}
    maxacc=0
    maxaccws=0
    #Holds the touple containing information of the model which performed best at certain number of features
    maxacct=(0,'0',0, touple)
    #Holds the touple containing information of model which performed best when all features were used    
    maxaccwst=(0,'0',0, touple)
    

    #Applying KNN
    for i in range(2,20):
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(clf, X_selected_s, results, cv=10, scoring='accuracy')
        if scores.mean()>maxacc:
            maxacct=(X_selected.shape[1], "KNN",scores.mean(), {"neighbours":i})
            maxacc=scores.mean()
        
        if flag:
            scores = cross_val_score(clf, X_poly_scale, results, cv=10, scoring='accuracy')
            temp=str(scores.mean())
            if scores.mean()>maxaccws:
                maxaccwst=("KNN",scores.mean(), {"neighbours":i})
                maxaccws=scores.mean()
            
    print("Accuracy WITH feature selection Using "+str(X_selected.shape[1])+" features and KNN:",maxacc)
    print("Accuracy WITHOUT feature selection and KNN:",maxaccws)

    #Naive Bayes
    clf = GaussianNB(priors=None)
    scores = cross_val_score(clf, X_selected, results, cv=10, scoring='accuracy')
    print("Naive Bayes Accuracy WITH "+str(X_selected.shape[1])+" features selected:",scores.mean())
    if scores.mean()>maxacc:
            maxacct=(X_selected.shape[1],"Naive Bayes",scores.mean(),{"priors":"None"})
            maxacc=scores.mean()
    if flag:       
        scores = cross_val_score(clf, X_poly, results, cv=10, scoring='accuracy')
        print("Naive Bayes Accuracy WITHOUT feature selection:",scores.mean())
        if scores.mean()>maxaccws:
                maxaccwst=("Naive Bayes",scores.mean(),{"priors":"None"})
                maxaccws=scores.mean()

    #Random Forest
    for nest in range(1,30):
        for mdp in range(1,30):
            forest = RandomForestClassifier(n_estimators=nest, random_state=10, max_depth=mdp, max_features=X_selected.shape[1])
            scores = cross_val_score(forest, X_selected, results, cv=10, scoring='accuracy')
            if scores.mean()>maxacc:
                maxacct=(X_selected.shape[1],"Random Forest",scores.mean(),{"n_estimators":nest,"max_depth":mdp})
                maxacc=scores.mean()
            if flag:
                forest = RandomForestClassifier(n_estimators=nest, random_state=10, max_depth=mdp, max_features=X_poly.shape[1])
                scores = cross_val_score(forest, X_poly, results, cv=10, scoring='accuracy')
                if scores.mean()>maxaccws:
                    maxaccwst=("Random Forrest",scores.mean(),{"n_estimators":nest,"max_depth":mdp})
                    maxaccws=scores.mean()

    print("Accuracy WITH feature selection Using "+str(X_selected.shape[1])+" features and Random Forest:",maxacc)
    print("Accuracy WITHOUT feature selection and Random Forest:",maxaccws)

    withs.append(maxacct)
    if flag:
        flag=False

print(withs)
print(maxaccwst)

# ### Now to the results

import matplotlib.pyplot as plt
from pandas import ExcelFile
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Results With Feature Selection
nfeatures=[item[0] for item in withs]
accuracies_selected=[item[2] for item in withs]
accuracies_selected=[float("{:.2f}".format(i*100)) for i in accuracies_selected]

plt.plot(nfeatures,accuracies_selected,'r') # 'r' is the color red
plt.xlabel('Number of Features')
plt.ylabel('Accuracies')
plt.title('Feature Selection')
plt.tight_layout()
plt.grid(alpha=0.8)
plt.xlim(10,110)
plt.ylim(60,100)
plt.figure(figsize=(70,80))

# ### As we can see in the graph that with the increase in number of selected features, our accuracy kept on increasing until we reach a certain point after which the increase in number of features not only increased the computational cost but also reduced the accuracy.
# 
# Now coming to all the Questions one by one.
# 
# #### 1- Do all features give better prediction? Or can we ignore some for better accuracy?
# 
# Answer: As clear from the data above, No. Only some of them do. We have maximum accuracy at almost a median number of features.
# 
# #### 2- How many features give us the best classification accuracy? Which model works best with the optimized set of features? Which hypermeters are best for each model we choose?
# 
# Answer: Check Output of Cell/Code Below

from operator import itemgetter
featureSelectionResults=sorted(withs,key=itemgetter(2),reverse=True)
print("Maxmimum Accuracy was at "+str(featureSelectionResults[0][0])+" features.\n")
print("The Model Which performed best Was "+str(featureSelectionResults[0][1]))
print("\nIt predicted with an Accuracy of "+"{:.2f}".format(featureSelectionResults[0][2]*100)+"%")
print("\nThe Hyperparameters for this model were "+str(featureSelectionResults[0][3]))

# ## Some Unasked Questions we all wanted answers of...
# 
# 
# ### The Output of the Code below gives all the models with their parameters which performed best at certain number of features selected

i=0
for item in withs:
    print("Features Selected: "+str(withs[i][0])+"\n")
    print("The Model Which performed best Was "+str(withs[i][1]))
    print("\nIt predicted with an Accuracy of "+"{:.2f}".format(withs[i][2]*100)+"%")
    print("\nThe Hyperparameters for this model were "+str(withs[i][3]))
    print("\n____________________________________________________________________________\n")
    i+=1

# ## But which model won WITHOUT the procedure of feature selection? Let's see the output of the cell below

print("The Model Which performed best Was "+str(maxaccwst[0]))
print("\nIt predicted with an Accuracy of "+"{:.2f}".format(maxaccwst[1]*100)+"%")
print("\nThe Hyperparameters for this model were "+str(maxaccwst[0][2]))
print("\n____________________________________________________________________________\n")

# # Conclusion
# 
# ### It's clear from the results above the feature selection improved our accuracy to a greater extent. Neither less, not too much features gave best results for our data.
# ### So too many features never assures good results and too few also don't! We have to find the balance.
