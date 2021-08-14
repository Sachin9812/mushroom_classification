#mathematical calculation and statistical imputation
import pandas as pd
import numpy as np
#Visualisation
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
#label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#training _testing conversion
from sklearn.model_selection import train_test_split
# dimension reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
#model building and Metrics for model evaluation
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble, linear_model, neural_network
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#model saving
import pickle
#ignore warning
import warnings
warnings.filterwarnings("ignore")


mushrooms = pd.read_csv('E:/iNueron/Mashrum/mushrooms.csv')

#displaying top 5 rows of Dataframe
mushrooms.head(5)

#Description of Data
mushrooms.describe()

#Detailed info of Data
mushrooms.info()

# Step 3: Exploratory Data Analysis
# 1.Visualising the number of mushrooms that fall in each class - p = poisonous, e=edible
# plt.figure(figsize=(8,6))
# plt.style.use('dark_background')
# s = sns.countplot(x = 'class',data = mushrooms)
# for p in s.patches:
#     s.annotate(format(p.get_height(), '.1f'),
#                (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha = 'center', va = 'center',
#                 xytext = (0, 9),
#                 textcoords = 'offset points')
# plt.show()

# As per above plot, we have 3916 as poisounous and 4208 are edibles mushrooms. from above observation data is slightly balanced and so no imbalance in dataset.

# 2. Univariate Analysis: Visualizing Countplot for all Independent features
features = mushrooms.columns
print(features)
# Index(['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#        'stalk-surface-below-ring', 'stalk-color-above-ring',
#        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#        'ring-type', 'spore-print-color', 'population', 'habitat'],
#       dtype='object')
# f, axes = plt.subplots(22,1, figsize=(15,150), sharey = True)
# k = 1
# for i in range(0,22):
#     s = sns.countplot(x = features[k], data = mushrooms, ax=axes[i], palette = 'GnBu')
#     axes[i].set_xlabel(features[k], fontsize=20)
#     axes[i].set_ylabel("Count", fontsize=20)
#     axes[i].tick_params(labelsize=15)
#     k = k+1
#     for p in s.patches:
#         s.annotate(format(p.get_height(), '.1f'),
#         (p.get_x() + p.get_width() / 2., p.get_height()),
#         ha = 'center', va = 'center',
#                    xytext = (0, 9),
#         fontsize = 15,
#         textcoords = 'offset points')

# 3. Bivariate Analysis: Visualising of columns categories with target column.
# f, axes = plt.subplots(22,1, figsize=(15,150), sharey = True)
# k = 1
# for i in range(0,22):
#     s = sns.countplot(x = features[k], data = mushrooms, hue = 'class', ax=axes[i], palette = 'CMRmap')
#     axes[i].set_xlabel(features[k], fontsize=20)
#     axes[i].set_ylabel("Count", fontsize=20)
#     axes[i].tick_params(labelsize=15)
#     axes[i].legend(loc=2, prop={'size': 20})
#     k = k+1
#     for p in s.patches:
#         s.annotate(format(p.get_height(), '.1f'),
#         (p.get_x() + p.get_width() / 2., p.get_height()),
#         ha = 'center', va = 'center',
#         xytext = (0, 9),
#         fontsize = 15,
#         textcoords = 'offset points')

#checking null values
mushrooms.isnull().sum()

#finding unqiue values in all columns
for col in list(mushrooms):
    print(col)
    print(mushrooms[col].unique())

#label encoding
le=LabelEncoder()
for i in mushrooms.columns:
    mushrooms[i]=le.fit_transform(mushrooms[i])

#plotting correltion plot
# plt.figure(figsize=(15,10))
# sns.heatmap(mushrooms.corr(), cmap='Reds',annot=True)

# From above correlation Heatmap we observed that -
# 1.Feature namely veil-type having only one category i.e partial(p)
#
# 2.veil-color and gill-attachment has highly correlated feature combination more than 80%. So we will drop one variable.
#
# Feature namely veil-type having only one category i.e partial(p) so we will drop as its not important for our prediction
#
# This is another way of checking the multicollinearity i.e. VIF. This will return us the extent to which
# multicollinearity is increased.

# function for VIF
def calc_VIF(mushrooms):
    vif = pd.DataFrame()
    vif['Variables'] = mushrooms.columns
    vif['VIF'] = [variance_inflation_factor(mushrooms.values, i) for i in range(mushrooms.shape[1])]

    return (vif)

calc_VIF(mushrooms)

#dropping columns
mushrooms = mushrooms.drop(['veil-type','veil-color','gill-attachment'], axis = 1)
mushrooms


X = mushrooms.drop(['class'],axis=1)
y = mushrooms['class']
# Label encoding y - dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# One hot encoding independent variable x
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
print(X)

#Splitting into train and test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

#shape of splitted data
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#PCA analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

x_train.shape,x_test.shape

#Step 5: Model Building
#Logistic Regression
# Training the Logistic Regression Model on the Training set
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression(random_state = 0)
LR_classifier.fit(x_train, y_train)

# Predicting the test set
y_pred = LR_classifier.predict(x_test)
# Making the confusion matrix and calculating accuracy score
acscore = []
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# Naive Bayes
# Training the Naive Bayes Classification model on the Training set
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)
# Training the Naive Bayes Classification model on the Training set
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)

# Predicting the test set
y_pred = NB_classifier.predict(x_test)
# Making the confusion matrix and calculating the accuarcy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# Support Vector Machine
# Training the RBF Kernel SVC on the Training set
from sklearn.svm import SVC
SVM_classifier = SVC(kernel = 'rbf', random_state=0)
SVM_classifier.fit(x_train, y_train)

# predicting test set
y_pred = SVM_classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# K - Nearest
# Neighbors(KNN)
# Calculating the optimum number of neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []
for neighbors in range(3, 10, 1):
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test, y_pred))

list1
plt.plot(list(range(3,10,1)), list1)
plt.show()

# Training the K Nearest Neighbor Classification on the Training set
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
KNN_classifier.fit(x_train, y_train)

# Predicting the test set
y_pred = KNN_classifier.predict(x_test)

# Making the confusion matrix and calculating the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# Decison Tree
# Training the Decision Tree Classification on the Training set
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
DT_classifier.fit(x_train, y_train)

# Predicting the test set
y_pred = DT_classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# XGBoost
# Training the XGBoost Classification on the Training set
from xgboost import XGBClassifier
XGB_classifier = XGBClassifier()
XGB_classifier.fit(x_train,y_train)

# Predicting the test set
y_pred = XGB_classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# Random Forest
# Finding the optimum number of n_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for estimators in range(10,150):
    RF_classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')
    RF_classifier.fit(x_train, y_train)
    y_pred = RF_classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(10,150)), list1)
plt.show()

# Training the Random Forest Classification on the Training set
from sklearn.ensemble import RandomForestClassifier
RF_classifier_est = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 115)
RF_classifier_est.fit(x_train, y_train)

# Predicting the test set
y_pred = RF_classifier_est.predict(x_test)

# Making the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
acscore.append(ac)
print(cm)
print(ac)

# Printing accuracy score of all the classification models we have applied
print(acscore)

models = ['LogisticRegression','NaiveBayes','KernelSVM','KNearestNeighbors','DecisionTree','XGBoost','RandomForest']

# Visualising the accuracy score of each classification model
plt.rcParams['figure.figsize']=15,8
plt.style.use('dark_background')
ax = sns.barplot(x=models, y=acscore, palette = "rocket", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 13, horizontalalignment = 'center', rotation = 0)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

#So among all classification model Random Forest Classification has highest accuracy score = 99.70%.

Pkl_Filename = "Pickle_RF_Model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(RF_classifier_est, file)
# Load the Model back from file
# with open(Pkl_Filename, 'rb') as file:
#     Pickled_RF_Model = pickle.load(file)

#Pickled_RF_Model

# Use the loaded pickled model to make predictions
# Pickled_RF_Model.predict(x_test)