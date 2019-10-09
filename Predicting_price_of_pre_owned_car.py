# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:35:40 2019

@author: Narendra Mall
"""

"""
Solution Conceptualization:
   - Identify if data is clean
   - Look for missing values
   - Identify variables influencing price and look for relationships among variables
      - Correlation, Box plot, scatter plots, count plot, crosstab table etc
   - Identify outliars
      - Central tendency measures, dispersion measures, boxplot, histogram etc.
   - Identify if categories with meagre frequencies can be combined    
   - Filter data based on logical checks
      - price, year of registartion, power
   - Reduced number of data
"""

#=====================================
# Predicting price of pre-owned Car
#=====================================

#first import necessary library 
#import pandas to generate and work on dataframe
import pandas as pd

#import numpy to do various calculation
import numpy as np

#import seaborn for visualisation
import seaborn as sns

#for partition the data
from sklearn.model_selection import train_test_split

#import library for logistic regression 
from sklearn.linear_model import LogisticRegression

#to look for performance importing accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Reading csv file
cars_data=pd.read_csv('cars_sampled.csv')

#creating copy
cars=cars_data.copy()

# Checking description of data
describe=cars.describe()

# structure of data
cars.info()

# summarising the numerical data
summary_num=cars.describe()

# summarising of categorical data
summary_cat=cars.describe(include='O')

#============================
# dropping unwanted column
#============================
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col, axis=1)# here axis=1 means column

##############################
### Removing duplicate record
##############################

cars.drop_duplicates(keep='first',inplace=True)
# so we saw that cars data decresed by 470

#############################
######### Data cleaning
#############################

# No of missing value in each column
cars.isnull().sum()

# variable yearOfResistration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()

sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
########### with the above plot we can see at some point there is lot of hike in price and at some of the places it is zero and 
#does make any sense so we have to clean the data
########### here we are considering <year of resistartion> b/w 1950 && 2018

# Variable price
price_count=cars['price'].value_counts() 
sns.distplot(cars['price'])
describe_price=cars.price.describe()
## from the above observation we can see that there is lot of price are outliar

#### outliar , Here we are considering price above 1,50,000 and less than 100 to be outliar
# lets have their count
sum(cars['price']>150000)
sum(cars['price']<100)

## coming to next variable powerPS
power_count=cars['powerPS'].value_counts().sort_index()
# here we can see there is lot of vehicle having power 0 that does not make any sense and there is also some of 
## them having very high value so all those data we will consider as outliar in this model

sns.distplot(cars['powerPS'])
describe_power=cars.powerPS.describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
# with the above graph we car refer that most of the car has less power and less price and that is fine also

sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
## here we are getting this range just by hit and trial 
## with keeping in mind that we don't loose much data at the same time

####################################
### Working range of data
#####################################

cars=cars[(cars.yearOfRegistration <= 2018) & (cars.yearOfRegistration >= 1950) & (cars.price >=100) & (cars.price <= 150000) & (cars.powerPS >= 10) & (cars.powerPS <= 500)]

# here almost 6760 data have been droped

# further to simplify: variable reduction
# combining yearOfResgistration and monthOfRegistration

cars['monthOfRegistration']/=12

# create new variable 'Age' 
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
describe_Age=cars.Age.describe()

# Drop yearOfRegistration and monthOfRegistration
cars=cars.drop(['yearOfRegistration','monthOfRegistration'],axis=1)

# visualisation of parameter

# Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

# price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

# powerPS 
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

# Age Vs price
sns.regplot(x='Age',y='price',scatter=True,data=cars)

# powerPS Vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)


# variable seller 
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)

# variable offerType
cars['offerType'].value_counts()
sns.countplot(x='abtest',data=cars)
# all cars have 'offer'=> Insignificant

# variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
# they are moreover equally distributed

sns.boxplot(x='abtest', y= 'price',data=cars)
# for every price value there is almost 50-50 distribution
# Does not affect price much so are considering it Insignificant

# Variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
# with the above boxplot we can see vehicleType have significant effect on price 

# variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
# here clearly we can see gearbox type making affect the price of vehicle

# variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price', data=cars)
# here cars distributed over many models
# so we are cosidering in our modelling

# variable kiometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True).sort_index()
sns.boxplot(x='kilometer',y='price',data=cars)
# Here we can see price does change on the change of kilometer
# so we will consider this feature


# variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
# with the help of boxplot we can see fuelType affect price

# Variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
# Here we can see car's price are distributed over brand
# consider in model

# variable notRepairedDamage
# yes- means its been rapaired
# no- means not been repaired after damage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
# As Expected, the cars which are not repaired having less price
# so we will consider this in our model


###############################
#Removing insignificant variable
##################################


# variable abtest
sns.countplot(x='abtest',data=cars)
sns.boxplot(x='abtest',y='price',data=cars)
# with the above plot we can see there is not much change in mean value of prive and count
# so we will not cosider this in our model
# by not using abtest featute model may loose accuracy but not very significant\

#############################
##### Removing insignificant variable
#######################################

col=['seller','offerType','abtest']
cars=cars.drop(col,axis=1)
cars_copy=cars.copy()


#####################################
################ correlation
######################################

cars_num=cars.select_dtypes(exclude=[object])
correlation=cars_num.corr()
round(correlation,3)
cars_num.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


"""
Now we are going to build a linear regression and random forest model 
on the two sets of data
1. data obtained by ommiting rows with any missing value
2. data obtained by imputing the missing values
"""

#############################
########## 1. Omitting missing value
#############################

cars_omit=cars.dropna(axis=0)

# converting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#############################
## Importing necessary libraries
##################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



#####################################
## Model building with omitted data
####################################

# separating input and output features
x1= cars_omit.drop(['price'],axis=1)
y1=cars_omit['price']

# plotting the variable price
prices=pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()
# here we can see that after taking log price data are in a bell shape form

# transforming price as a logarithmic value
y1=np.log(y1)

# splitting the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(x1,y1,test_size=0.3,random_state=0)

#########################################
######## Linear Regression with omitted data
#######################################

# setting intercept as true
lgr= LinearRegression(fit_intercept=True)

# fit the model
model_lin1=lgr.fit(X_train,Y_train)

# predicting model on test set
cars_predictions_lin1=lgr.predict(X_test)

# computing MSE and RMSE
lin_mse1=mean_squared_error(cars_predictions_lin1,Y_test)
lin_rmse1=np.sqrt(lin_mse1)

# R squared value
r2_lin_test1=model_lin1.score(X_test,Y_test)
r2_lin_train1=model_lin1.score(X_train,Y_train)


# Regression diagnostics - residual plot analysis
residuals1=Y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,fit_reg=False)

residuals1.describe()


###########################################
### Random Forest with omitted data
###########################################

# Model Parameter
rf= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,
                          min_samples_leaf=4,random_state=1)

# Model using rf
model_rf1=rf.fit(X_train,Y_train)

# predicting model on test set
cars_predictions_rf1=rf.predict(X_test)

# computing MSE and RMSE
rf_mse1=mean_squared_error(Y_test,cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)


##################################################
######### Model Building with Imputed data
#################################################

cars_imputed= cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))

# check for null
cars_imputed.isnull().sum()

# converting categorical variable to dummy variable
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)


### separating input and output feature
x2= cars_imputed.drop(['price'],axis=1)
y2=cars_imputed['price']

# take the natural log of y2 as taken before
y2= np.log(y2)

#=================================================
# Linear Regression with imputed data
#==================================================

# Splitting data into test and train
X_train1,X_test1,Y_train1,Y_test1=train_test_split(x2,y2,test_size=0.3,random_state=1)

# setting intercept as true
lgr2= LinearRegression(fit_intercept=True)

#fit the model
model_lin2=lgr2.fit(X_train1,Y_train1)


# predicting the model on test set
cars_predictions_lin2=lgr2.predict(X_test1)


# computing MSE and RMSE
lin_mse2= mean_squared_error(Y_test1,cars_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)



#======================================================
# Random Forest with imputed data
#========================================================

# Model parameters
rf2= RandomForestRegressor(n_estimators=100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

# fit model
model_rf2=rf2.fit(X_train1,Y_train1)

# predict modelon test set
cars_predictions_rf2=rf2.predict(X_test1)

# Computing rmse and mse
rf_mse2=mean_squared_error(Y_test1,cars_predictions_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)
# so this one is most accurate model
# here we are getting the rmse value= 0.4887 


# R squared value
r2_rf_test2=model_rf2.score(X_test1,Y_test1)
r2_rf_train2=model_rf2.score(X_train1,Y_train1)






