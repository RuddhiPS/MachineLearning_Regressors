import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data

data.info()

data.describe()

data.isnull().sum()

data.drop(['car name','origin','model year'],axis=1,inplace=True)

data.head()

data.describe()

data.dtypes

data['horsepower'].unique()

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

data

data.dtypes

data['horsepower'].isnull()

data = data[data['horsepower'].notna()]

data['horsepower'].isnull().sum()

data

features=data[['cylinders','displacement','horsepower','weight','acceleration']]

corr_matr=features.corr()

print(corr_matr)

plt.figure(figsize=(8,6))
import seaborn as sns
sns.heatmap(corr_matr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

data.drop(['displacement'],axis=1,inplace=True)#dropping displacement as it is highly correlated

data

data=data.sample(frac=1,random_state=42).reset_index(drop=True)

X=data.iloc[:,1:]

X

y=data.iloc[:,1]

y

X.shape[0]

split_id=int(0.7*X.shape[0])

X_train=X.iloc[:split_id,:]
X_test=X.iloc[split_id:,:]
y_train=y.iloc[:split_id]
y_test=y.iloc[split_id:]

X_train.shape

y_train.shape

y_test.shape

dtree=DecisionTree(criterion='mse')

dtree.fit(X_train,y_train)

y_pred=dtree.predict(X_test)

print("Accuracy of our model: ",accuracy(y_pred,y_test))

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy of sklearn model: ",accuracy_score(y_test,y_pred))

