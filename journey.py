import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


titanic_df = titanic_df.drop(['PassengerId'], axis=1)



def changeFeatureDataType(dataframe):
	sex_dummies= pd.get_dummies(dataframe['Sex'])
	sex_dummies.drop(['female'], axis=1, inplace=True)
	dataframe = dataframe.join(sex_dummies)
    # convert the Embarked values to numeric values s=0, c=1, q=2
	embark_dummies= pd.get_dummies(dataframe['Embarked'])
	embark_dummies.drop(['S'], axis=1, inplace=True)
	dataframe = dataframe.join(embark_dummies)
	
	dataframe = dataframe.drop(['Embarked','Sex'], axis=1)
	return dataframe

titledict = {"Dr"   : "Mr",
             "Col"  : "Officer",
             "Mlle" : "Miss",
             "Major": "Officer",
             "Lady" : "Royal",
             "Dona" : "Royal",
             "Don"  : "Royal",
             "Mme"  : "Mrs",
             "the Countess": "Royal",
             "Jonkheer": "Royal",
             "Capt" : "Officer",
             "Sir"  : "Mr"
             }
			 
def hadtitle(dataframe):
	dataframe['Title']= dataframe.Name.str.replace('(.*, )|(\\..*)', '')
	dataframe = dataframe.drop(['Name'], axis=1)
	for key,val in titledict.items():
		dataframe.loc[dataframe["Title"]==key, "Title"] = val
	return dataframe

def cabinside(dataframe):
	dataframe["CabinSide"] = "Unknown"
	dataframe.loc[pd.notnull(dataframe["Cabin"]) & dataframe["Cabin"].str[-1].isin(["1", "3", "5", "7", "9"]),"CabinSide"] = "Starboard"
	dataframe.loc[pd.notnull(dataframe["Cabin"]) & dataframe["Cabin"].str[-1].isin(["0", "2", "4", "6", "8"]),"CabinSide"] = "Port"
	return dataframe
	
def cabindeck(dataframe):
	dataframe["Deck"] = "Unknown"
	dataframe.loc[pd.notnull(dataframe["Cabin"]), "Deck"] = dataframe["Cabin"].str[0]
	dataframe.loc[pd.notnull(dataframe["Cabin"])& (dataframe.Cabin.str.len() == 5), "Deck"] = dataframe["Cabin"].str[2]
	dataframe.loc[dataframe["Deck"] == 'T',"Deck"] = "Unknown"
	return dataframe
	
def handleMissingValues(dataframe):
    dataframe["Age"]   = dataframe["Age"].fillna(dataframe["Age"].median())
    dataframe["Fare"]   = dataframe["Fare"].fillna(dataframe["Fare"].median())
    return dataframe
	
def familisize(dataframe):
	dataframe["FamilySize"] = dataframe.Parch + dataframe.SibSp + 1
	return dataframe
	
def removeUnwantedfeatures(dataframe):
    dataframe = dataframe.drop(['Ticket'], axis=1)
    return dataframe
	
def transform_dataframe(df):
	df_transformed = handleMissingValues(df)
	df_transformed = changeFeatureDataType(df_transformed)
	df_transformed=hadtitle(df_transformed)
	df_transformed=cabinside(df_transformed)
	df_transformed=cabindeck(df_transformed)
	df_transformed=familisize(df_transformed)
	df_transformed = removeUnwantedfeatures(df_transformed)

	return df_transformed
	
titanic_df=transform_dataframe(titanic_df)
test_df=transform_dataframe(test_df)
print(titanic_df.head(20))

"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


k_range = range(1, 31)
param_grid = dict(n_neighbors=list(k_range),weights = ["uniform", "distance"])
"""