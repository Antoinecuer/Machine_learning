import csv
import math
import urllib.request
import matplotlib.pyplot as plt
import pandas as pandas
import numpy
import sklearn.preprocessing as preprocessing
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

#read data into iterable
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
names = ['Age', 'Workclass', 'fnlwgt', 'education', 'educationnum', 'maritalstatus', 'occupation', 'relationship', 'race','sex','capitalgain','capitalloss','hoursperweek','nativecountry','MoreThan50k']
categorical_data=['Workclass','education','maritalstatus', 'occupation', 'relationship', 'race','sex','nativecountry','MoreThan50k']
with urllib.request.urlopen(target_url) as url:
	#s = url.read()
	data=pandas.read_csv(url,names=names,index_col=False)


data=data[data.nativecountry != ' ?']
data=data[data.Workclass != ' ?']
data=data[data.occupation != ' ?']
#print(data.relationship.value_counts())
for a in categorical_data:
	j=0
	for i in data[a].unique():
		data[a].loc[data[a]==i]=j
		j+=1



"""
sns.countplot(y='education', hue='MoreThan50k', data=data,)
sns.plt.show()
"""
"""
g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
g = g.map(sns.boxplot, 'MoreThan50k', 'educationnum')
sns.plt.show()
"""
"""
sns.violinplot(x='sex', y='educationnum', hue='MoreThan50k', data=data, split=True, scale='count')
sns.plt.show()
"""
"""
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == numpy.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

encoded_data, _ = number_encode_features(data)	
print(encoded_data)
"""
"""
data['Gender'] = data['sex'].map( {' Female': 0, ' Male': 1} ).astype(int)
data=data.drop('sex',1)

data['White']= data['race'].map( {' Black': 0,' White': 1,' Asian-Pac-Islander': 0,' Amer-Indian-Eskimo': 0,' Other': 0} ).astype(int)
data['Black']= data['race'].map( {' Black': 1,' White': 0,' Asian-Pac-Islander': 0,' Amer-Indian-Eskimo': 0,' Other': 0} ).astype(int)
data['Asian']= data['race'].map( {' Black': 0,' White': 0,' Asian-Pac-Islander': 1,' Amer-Indian-Eskimo': 0,' Other': 0} ).astype(int)
data['Indian']= data['race'].map( {' Black': 0,' White': 0,' Asian-Pac-Islander': 0,' Amer-Indian-Eskimo': 1,' Other': 0} ).astype(int)
data['Other']= data['race'].map( {' Black': 0,' White': 0,' Asian-Pac-Islander': 0,' Amer-Indian-Eskimo': 0,' Other': 1} ).astype(int)
data=data.drop('race',1)

data['American']=0
data['American'].loc[data['nativecountry']==' United-States']=1
data=data.drop('nativecountry',1)

data['Rich'] = data['MoreThan50k'].map( {' <=50K': 0, ' >50K': 1} ).astype(int)
data=data.drop('MoreThan50k',1)
"""

"""
names=['Age','fnlwgt','educationnum','capitalgain','capitalloss','hoursperweek','gender','white','Black','Asian','Indian','Other','American','Rich']
correlations = data.corr()
"""

"""
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

fig = plt.figure(figsize=(20,15))
cols = 5
rows = math.ceil(float(data.shape[1]) / cols)
for i, column in enumerate(data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data.dtypes[column] == numpy.object:
        data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()
"""