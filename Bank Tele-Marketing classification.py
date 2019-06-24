import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DataFrame, concat
import seaborn as sns
from datetime import datetime as dt

%matplotlib inline
import statsmodels as sm

import pandas as pd

dataset = pd.read_csv("bank-full.csv", sep=';',header='infer')
ataset = pd.read_csv("bank-full.csv", sep=';',header='infer')

dataset["month"] = dataset["month"].str.capitalize()
# make it a datetime so that we can sort it: 
# use %b because the data use the abbriviation of month

dataset["month"] = pd.to_datetime(dataset.month, format='%b', errors='coerce').dt.month

#because dataset does not ahave a year column
dataset['year'] = 2011
dataset['date'] = dataset['day'].map(str) + '-' + dataset['month'].map(str) + '-' + dataset['year'].map(str)


#dataset = dataset.sort_values(by="month")

del dataset['day'] 
del dataset['month'] 
del dataset['year'] 

#change index of columns
columnsTitles = ['date','age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous',
       'poutcome', 'y']
dataset=dataset.reindex(columns=columnsTitles)
#rename
dataset=dataset.rename(columns = {'date':'date(D-M)'})
#check na values
dataset.isnull().values.any()


#no missing values
#summary of dataset
dataset.describe()

#change name of dataset column
dataset =dataset.rename(columns={ dataset.columns[15]: "subscribed" })

dataset.head()
dataset.columns

#for using during OneHotEcoding
dsub = dataset.iloc[:,0:15]
dsub.dtypes
dsub['date(D-M)'] = pd.to_datetime(dsub['date(D-M)'])
categorical_feature_mask_x = dsub.dtypes==object
categorical_cols_x = dsub.columns[categorical_feature_mask_x].tolist()
#categorical variables
#covert object to date time


dataset['date(D-M)'] = pd.to_datetime(dataset['date(D-M)'])


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

dataset.dtypes
# Categorical boolean mask to take al objects
categorical_feature_mask = dataset.dtypes==object

# filter categorical columns using mask and turn it into a list
categorical_cols = dataset.columns[categorical_feature_mask].tolist()
#label encode categorical variables
dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col: labelencoder_X.fit_transform(col))
dataset[categorical_cols].head(10)

x = dataset.iloc[:, 0:15]
x.columns

x = x.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,13,14]]
#"PDAYS" says number of days that passed after the client was last contacted 
#from previous campaign 
#there is is -1 value which says client was not contacted before this can lead 
#to be dominating the prediction model so it's better to remove it

#"duration" of call is in seconds so if we keep this in prediction aur model will dominate 
#towards more number of seconds so to get realistic predictive model
#it is important to remove DURATION


y = dataset.iloc[:,15]

#sparse : boolean, default=True Will return sparse matrix if set True else will return an array.
#object is only fitted to the categorical values
# filter categorical columns using mask and turn it into a list
#sparse Flase output an array not an matrix
print(dsub.dtypes==object)

#it handles the dummy variable trap 
#so no need of one hot encoding
x = pd.get_dummies(x, columns=['job', 'marital','education', 'default','housing','loan','contact', 'poutcome'], drop_first=True)
del x['date(D-M)']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#scale the data for avoiding dominating features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras will built the deep neural network based on tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# Adding the input layer and the first hidden layer
#outputdim=choosing number of hidden nodes 28 independent variable + dependent variable divide by 2
#init= initialize weights to small number close to zero
#input_dim= number of dependent variables = 28
model.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 28))

# Adding the second hidden layer
model.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))

# Adding the output layer
#if dependent variable has more than 2 category use softmax activation function in output
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
#if dependent variable has more than 2 category LOSS FUNCTION Will be categorical_crossentropy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# the model validated on this telemarketing dataset will help bank to perform the same model on different customers of the bank
#as the y_pred results in the probability of the customer subscribing to the bank term deposit or not
# the bank will get the ranking of the customer for example takig the 10% of customers not subscribing to term deposit
# this helps bank to measure and to overcome the problem of customer not subscribing 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

#least interested customer

# 0 didnt subscribed to the bank deposit
# 1 subscribed to the bank deposit

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#from CM
(7865 + 179)/9043

#Accuracy
0.8895278115669578
#Percentage
88.95%
