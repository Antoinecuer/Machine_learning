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


def normalize(feat):
    mean = full[feat].mean()
    stdv = full[feat].std()
    for df in [titanic_df,test_df]:
        df[feat + "_norm"] = (df[feat] - mean) / stdv
def changeFeatureDataType(dataframe):
	sex_dummies= pd.get_dummies(dataframe['Sex'])
	sex_dummies.drop(['male'], axis=1, inplace=True)
	dataframe = dataframe.join(sex_dummies)
    
	embark_dummies= pd.get_dummies(dataframe['Embarked'])
	embark_dummies.columns = ['EmbarkedC','EmbarkedQ','EmbarkedS']
	embark_dummies.drop(['EmbarkedS'], axis=1, inplace=True)
	dataframe = dataframe.join(embark_dummies)
	
	pclass_dummies= pd.get_dummies(dataframe['Pclass'])
	pclass_dummies.columns = ['Class_1','Class_2','Class_3']
	pclass_dummies.drop(['Class_3'], axis=1, inplace=True)
	dataframe = dataframe.join(pclass_dummies)
	
	CabinSide_dummies= pd.get_dummies(dataframe['CabinSide'])
	CabinSide_dummies.drop(['Unknown'], axis=1, inplace=True)
	dataframe = dataframe.join(CabinSide_dummies)
	
	Deck_dummies= pd.get_dummies(dataframe['Deck'])
	Deck_dummies.columns = ['Deck_A','Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Unknown']
	#Deck_dummies.drop(['Unknown'], axis=1, inplace=True)
	dataframe = dataframe.join(Deck_dummies)
	
	Title_dummies= pd.get_dummies(dataframe['Title'])
	Title_dummies.drop(['Mr'], axis=1, inplace=True)
	dataframe = dataframe.join(Title_dummies)
	
	dataframe = dataframe.drop(['Sex','Pclass','CabinSide','Title','Embarked'], axis=1)
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
             "Sir"  : "Mr",
			 "Ms"   : "Miss"
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
    dataframe = dataframe.drop(['Cabin','Ticket'], axis=1)
    return dataframe
	
def transform_dataframe(df):
	df_transformed = handleMissingValues(df)
	df_transformed=hadtitle(df_transformed)
	df_transformed=cabinside(df_transformed)
	df_transformed=cabindeck(df_transformed)
	df_transformed=familisize(df_transformed)
	df_transformed = changeFeatureDataType(df_transformed)
	df_transformed = removeUnwantedfeatures(df_transformed)

	return df_transformed
	
titanic_df=transform_dataframe(titanic_df)
test_df=transform_dataframe(test_df)

X_train = titanic_df.drop("Survived",axis=1)


from sklearn import linear_model
"""
corr_mat = np.corrcoef(titanic_df.values.T)

ax = sns.heatmap(corr_mat, annot=True, fmt='.2f',
                 xticklabels=titanic_df.columns, yticklabels=titanic_df.columns,
                )

_ = (ax.set_title('Correlation Matrix'))

plt.show()

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print('Logistic regression')
print(logreg.score(X_train, Y_train))

from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print('SVC')
print(svc.score(X_train, Y_train))

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_RandomForest = random_forest.predict(X_test)
print('RandomForestClassifier')
print(random_forest.score(X_train, Y_train))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print('KNeighborsClassifier')
print(knn.score(X_train, Y_train))

coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print(coeff_df)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_RandomForest
    })
submission.to_csv('titanic.csv', index=False)
"""
