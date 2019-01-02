import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont


dataset = pd.read_csv('train.csv')
df = pd.read_csv('train.csv')
df.drop('Id',axis =1 ,inplace = True)
#dataset.drop(['month','day'],axis =1 ,inplace = True)
#handle dummy trap also
#df_2 = pd.get_dummies(dataset,columns=['month','day'],drop_first = True)
#df_3 = pd.get_dummies(df,columns=['month','day'],drop_first = True)


replace_map = {'month': {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                                  'may': 5, 'jun': 6, 'jul': 7 , 'aug': 8 , 'sep': 9,'oct': 10,'nov': 11,'dec':12}}



replace_map1 = {'day': {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4,
                                  'fri': 5, 'sat': 6, 'sun': 7 }}                                  
dataset_replace = dataset.copy()                                 
dataset_replace.replace(replace_map, inplace=True)
dataset_replace.replace(replace_map1, inplace=True)

X  = dataset_replace.iloc[:,0:12].values
y = dataset_replace.iloc[:,12 ].values

from sklearn.feature_selection import RFE
model = ExtraTreesRegressor()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.15, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=2)
regressor.fit(X_train,y_train)


     
print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)

df_replace = df.copy()                                 
df_replace.replace(replace_map, inplace=True)
df_replace.replace(replace_map1, inplace=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = sc.transform(X)
y = sc.fit_transform(y)
y = sc.transform(y)


"""
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
X.shape
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, encoded2)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
"""
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
X_new = SelectKBest(chi2, k=6).fit_transform(X, encoded2)
X_new.shape



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf = clf.fit(X, encoded2)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape  



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(X_train,y_train)

from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)

lab_enc1 = preprocessing.LabelEncoder()
encoded1 = lab_enc1.fit_transform(y_test)

lab_enc2 = preprocessing.LabelEncoder()
encoded2 = lab_enc2.fit_transform(y)

encoded2=lab_enc2.inverse_transform(encoded2)

# Predicting values using our trained model
y_pred = regressor.predict(X_test)
"""
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, encoded)
"""
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
"""
from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)

from sklearn import preprocessing
from sklearn import utils
lab_enc1 = preprocessing.LabelEncoder()
encoded1 = lab_enc1.fit_transform(y_test)
"""
from sklearn.preprocessing import scale
y_scale = scale(y)
mean_of_array = y.mean(axis=0)
std_of_array = y.std(axis=0)
y_original = (y_scale * std_of_array) + mean_of_array



y_pred=regressor.predict(X_test)


X_test = df.iloc[:,:].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_opt, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)
# Fitting Polynomial Regression to the dataset
 

np.savetxt('y_pred_1.csv',y_pred ,delimiter=',')



 