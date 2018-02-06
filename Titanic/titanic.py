import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from pandas import Series,DataFrame


data_train = pd.read_csv("train.csv")

data_train.info()

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
data_train = pd.read_csv("train.csv")

data_train.info()
data_train.describe()

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')# bar 
plt.title(u"Survived") # title
plt.ylabel(u"people")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"people")
plt.title(u"class")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")
plt.grid(b=True, which='major', axis='y') 
plt.title(u"Survived")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"age and class")
plt.legend((u'first class', u'2 class',u'3 class'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Embarked")
plt.ylabel(u"people")  
#plt.show()

#print(data_train.Cabin.value_counts())

def set_missing_ages(df):

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y
    y = known_age[:, 0]

    # X
    X = known_age[:, 1:]


    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)


    predictedAges = rfr.predict(unknown_age[:, 1::])


    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
