#importing libraries
import numpy
import pandas
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
#reading test and train data set
dataframe = pandas.read_csv("train.csv")
df =   pandas.read_csv("test.csv")
df.drop('Id',axis =1 ,inplace = True)


dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
df.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

#visualisation
print("Head:", dataframe.head())
print("Statistical Description:", dataframe.describe())
print("Correlation:", dataframe.corr(method='pearson'))
# 5,9,10
dataset = dataframe.values
X = dataset[:,0:12]
y= dataset[:,12]

X_test = df.iloc[:,:].values

"""
#Feature Selection
model = ExtraTreesRegressor()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
"""
#plot
plt.hist((dataframe.area))
dataframe.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)

#visualisation
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)

#split the dataset into test and train set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#DecisionTreeRegressor is giving best prediction
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=16)
regressor.fit(X_train,y_train)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 550, random_state = 42)
regressor.fit(X_train, y_train)


#predict on test test
y_pred = regressor.predict(X_test)

#calculating RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)


#converting to .csv file
numpy.savetxt('y_pred_1.csv',y_pred ,delimiter=',')

"""
# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Burnt Forest(Regression Model)')
plt.xlabel('bahut hai')
plt.ylabel('area')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""