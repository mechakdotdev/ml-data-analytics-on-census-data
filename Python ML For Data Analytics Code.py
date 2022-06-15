# Python Data Plotting for Census Data

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from scipy.stats import ttest_ind, ttest_rel
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#redefine missing values
missing_values = [" ?", "n/a","--", "nan"]
data = pd.read_csv("../input/thedata/adult.csv", na_values = missing_values)
data[data == '?'] = np.nan #helps machine realise that ? values are nan
for col in ['work_class', 'occupation', 'nativecountry']:
    data[col].fillna(data[col].mode()[0], inplace=True) #replace missing values with mode

X = data.drop(['under_over'], axis=1)

y = data['under_over']
    
#separate data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#transform any categorical data
categorical = ['work_class', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'nativecountry']
for feature in categorical:
        encoder = preprocessing.LabelEncoder()
        X_train[feature] = encoder.fit_transform(X_train[feature])
        X_test[feature] = encoder.transform(X_test[feature])
        
#scale the features
sk_scaler = StandardScaler()
X_train = pd.DataFrame(sk_scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(sk_scaler.transform(X_test), columns = X.columns) 


accuracyList = []
#logistic regression model using all the features
regression = LogisticRegression()
regression.fit(X_train, y_train)
y_predict = regression.predict(X_test)
theAccuracy = accuracy_score(y_test, y_pred)
print('Logistic regression:',  theAccuracy)
accuracyList.append(theAccuracy)

#below is the KNN portion
knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knnClassifier.fit(X_train, y_train)

y_predict = knnClassifier.predict(X_test)
theAccuracy = accuracy_score(y_test, y_pred)
print('K nearest neighbour:',  theAccuracy)
accuracyList.append(theAccuracy)

#below is the decision tree
dtClassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtClassifier.fit(X_train, y_train)

y_predict = dtClassifier.predict(X_test)
theAccuracy = accuracy_score(y_test, y_pred)
print('Decision tree:',  theAccuracy)
accuracyList.append(theAccuracy)

rfClassifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
rfClassifier.fit(X_train, y_train)

y_predict = rfClassifier.predict(X_test)
theAccuracy = accuracy_score(y_test, y_pred)
print('Random forest:',  theAccuracy)
accuracyList.append(theAccuracy)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#evaluating the algorithms
axisY =['Logistic Regression',
        'K-Neighbors Classifier',
        'Decision Tree Classifier',
        'Random Forest Classifier']
axisX=accuracyList
sns.barplot(x=axisX,y=axisY)
plt.xlabel('Accuracy')

#data.sample(10)
#data.isna().values.any() #check if any values are missing (result returns false as it doesn't recognise ? as missing values)
#data.shape #find the shape of the dataset (result: 32651 instances split into 15 features)

#data.info() # to check data types and other info; this tells us there are 5 numerical features and 9 categorical features

#data["age"].hist(figsize=(10,10))
#plt.xlabel('Age')
#plt.ylabel('Count')
#plt.show()

#data["fnlwgt"].hist(figsize=(10,10))
#plt.xlabel('Final Weight')
#plt.ylabel('Count')
#plt.show()

#data["education_num"].hist(figsize=(10,10))
#plt.xlabel('Educational Number')
#plt.ylabel('Count')
#plt.show()

#data["capital_gain"].hist(figsize=(10,10))
#plt.xlabel('Capital Gain')
#plt.ylabel('Count')
#plt.show()

#data["capital_loss"].hist(figsize=(10,10))
#plt.xlabel('Capital Loss')
#plt.ylabel('Count')
#plt.show()

#data["hours_pw"].hist(figsize=(10,10))
#plt.xlabel('Hours worked (per week)')
#plt.ylabel('Count')
#plt.show()



#plt.figure(figsize=(15,9))
#x = sns.countplot(x="work_class", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="education", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="marital_status", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="occupation", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="relationship", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="sex", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="race", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="nativecountry", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="under_over", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#x = sns.countplot(x="sex", data=data)
#for p in x.patches:
#    height = p.get_height()
#    x.text(p.get_x()+p.get_width()/2.,
#            height + 3,
#            '{:1.2f}'.format((height/total)*100),
#            ha="center") 
#plt.show()

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["age"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["work_class"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["education"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["education_num"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["marital_status"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["occupation"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["relationship"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["race"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#fig = plt.figure(figsize=(15,9))
#crosstab2=pd.crosstab(data["sex"],data["under_over"], normalize='index', margins=False)*100
#print(crosstab2)
#crosstab2.plot(kind = 'bar', stacked = True)

#plt.figure(figsize=(12, 8))
#sns.boxplot(x="fnlwgt", y="under_over", data=data)
#plt.show()

#plt.figure(figsize=(12, 8))
#sns.boxplot(x="capital_gain", y="under_over", data=data)
#plt.show()

#plt.figure(figsize=(12, 8))
#sns.boxplot(x="capital_loss", y="under_over", data=data)
#plt.show()

#plt.figure(figsize=(12, 8))
#sns.boxplot(x="hours_pw", y="under_over", data=data)
#plt.show()


#box plots to identify outliers
#num_feat = data.select_dtypes(include=['int64']).columns
#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.boxplot(data[num_feat[i]])
#    plt.title(num_feat[i],color="g",fontsize=22)
#    plt.yticks(fontsize=15)
#    plt.xticks(fontsize=15)
#plt.show()

#removing outliers through winsorization
#from scipy.stats.mstats import winsorize
#data["age"]           = winsorize(data["age"],(0,0.15))
#data["fnlwgt"]        = winsorize(data["fnlwgt"],(0,0.15))
#data["capital_gain"]  = winsorize(data["capital_gain"],(0,0.099))
#data["capital_loss"]  = winsorize(data["capital_loss"],(0,0.099))
#data["hours_pw"]      = winsorize(data["hours_pw"],(0.12,0.18))

#plt.rcParams['figure.figsize'] = (25,7)

#baslik_font = {'family':'arial','color':'red','weight':'bold','size':25}

#col_list=['age',"fnlwgt",'capital_gain', 'capital_loss', 'hours_pw']

#for i in range(5):
#    plt.subplot(1,5,i+1)
#    plt.boxplot(data[col_list[i]])
#    plt.title(col_list[i],fontdict=baslik_font)

#plt.show()

#data.head(10)

#numerical = ['int64']
#numericData = data.select_dtypes(include=numerical) #organise numerical data to be parsed separately

#categorical = ['object']
#categoricData = data.select_dtypes(include=categorical) #organise categorical data to be parsed separately

#print(numericData.describe()) #describe numerical dataset and return min/max/std etc
#print("Mode", categoricData.mode()) #display mode (most frequent attributes)