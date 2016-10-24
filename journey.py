import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
"""
import seaborn as sns
sns.set_style('whitegrid')
"""

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


#titanic_df = titanic_df.drop(['PassengerId'], axis=1)


		

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
	
	dataframe = dataframe.drop(['Sex','Pclass','CabinSide','Title','Embarked','Deck'], axis=1)
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

firstTime=1
def transform_dataframe(df):
	df_transformed = handleMissingValues(df)
	df_transformed = hadtitle(df_transformed)
	df_transformed = cabinside(df_transformed)
	df_transformed = cabindeck(df_transformed)
	df_transformed = familisize(df_transformed)
	df_transformed = changeFeatureDataType(df_transformed)
	df_transformed = removeUnwantedfeatures(df_transformed)

	return df_transformed
	
def normalization_dataframe(train_df,test_df):
	for column in train_df:
		if(column != "Survived"):
			if(column != "PassengerId"):
				test_df[column] = (test_df[column] - train_df[column].mean()) / train_df[column].std()
				train_df[column] = (train_df[column] - train_df[column].mean()) / train_df[column].std()

	return train_df,test_df

titanic_df=transform_dataframe(titanic_df)
test_df=transform_dataframe(test_df)
titanic_df,test_df=normalization_dataframe(titanic_df,test_df)


from sklearn import linear_model



"""
corr_mat = np.corrcoef(titanic_df.values.T)

ax = sns.heatmap(corr_mat, annot=True, fmt='.2f',
                 xticklabels=titanic_df.columns, yticklabels=titanic_df.columns,
                )

_ = (ax.set_title('Correlation Matrix'))

plt.show()
"""

X_trainf = titanic_df.drop("Survived",axis=1)
Y_trainf = titanic_df["Survived"]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_trainf, Y_trainf)

features = pd.DataFrame()
features['feature'] = X_trainf.columns
features['importance'] = clf.feature_importances_

model = SelectFromModel(clf, prefit=True)
X_trainf = model.transform(X_trainf)
test_df = model.transform(test_df)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X_trainf, Y_trainf, test_size=0.4, random_state=0)

from sklearn.model_selection import cross_val_score
from sklearn import svm

best_score_SVC=0
best_param_SVC=[0,0]
regularisation=[0.01,0.05,0.1,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.7,0.75,1,1.5,2,10]
kern=["linear","poly","rbf","sigmoid"]
for i in kern:
	for j in regularisation:
		clf = svm.SVC(C=j, kernel=i)
		scores = cross_val_score(clf, X_train, y_train, cv=20)
		if(scores.mean()>best_score_SVC):
			best_score_SVC=scores.mean()
			best_param_SVC=i,j
		print("SVM mean with kernel:",i,"regularisation:",j, scores.mean())
		print("------------------------")

print("SVC")
print(best_param_SVC, best_score_SVC)
print("--------------------------")


penalty=["l1","l2"]

best_score_logistic=0
best_param_logistic=[0,0]
for i in penalty:
	for j in regularisation:
		logreg = LogisticRegression(penalty=i,C=j)
		scores = cross_val_score(logreg, X_train, y_train, cv=20)
		if(scores.mean()>best_score_logistic):
			best_score_logistic=scores.mean()
			best_param_logistic=i,j
		#print("Logistic regression mean with penalty:",i,"regularisation:",j, scores.mean())
		#print("------------------------")
print("Logistic regression")
print(best_param_logistic,best_score_logistic)
print("--------------------------")
from sklearn.ensemble import RandomForestClassifier

estimator=[]
for i in range(15,25):
	estimator.append(5*i)
critere=["gini","entropy"]

best_score_RFC=0
best_param_RFC=[0,0]
for i in critere:
	for j in estimator:
		random_forest = RandomForestClassifier(n_estimators=j,criterion=i)
		scores = cross_val_score(random_forest, X_train, y_train, cv=20)
		if(scores.mean()>best_score_RFC):
			best_score_RFC=scores.mean()
			best_param_RFC=i,j

print("RandomForestClassifier")
print(best_param_RFC,best_score_RFC)
print("--------------------------")


from sklearn.neighbors import KNeighborsClassifier

n_neighbors=[]
for i in range(1,10):
	n_neighbors.append(i)

weights=["uniform","distance"]

best_score_knn=0
best_param_knn=[0,0]
for i in weights:
	for j in n_neighbors:
		knn = KNeighborsClassifier(n_neighbors = j,weights=i)
		scores = cross_val_score(knn, X_train, y_train, cv=20)
		if(scores.mean()>best_score_knn):
			best_score_knn=scores.mean()
			best_param_knn=i,j

print("KNeighborsClassifier")
print(best_param_knn,best_score_knn)
print("--------------------------")


coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print(coeff_df)

"""
final_SVC = svm.SVC(C=best_param_SVC[1], kernel=best_param_SVC[0])
final_SVC.fit(X_trainf, Y_trainf)
Y_pred_SVC = final_SVC.predict(test_df.drop(['PassengerId'], axis=1))
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_SVC
    })
submission.to_csv('titanic.csv', index=False)

"""
