import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tree.utils import preprocessing

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
X,y=preprocessing(X,y)

X

y

for criteria in ["mse"]:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))

# Test case 2
# Real Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")
X,y=preprocessing(X,y)

y


for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision for Class-{cls}: ", precision(y_hat, y, cls))
        print(f"Recall for Class-{cls}: ", recall(y_hat, y, cls))

# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")
X,y=preprocessing(X,y)

for criteria in ["entropy", "gini_index"]:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Precision for Class-{cls}: ", precision(y_hat, y, cls))
        print(f"Recall for Class-{cls}: ", recall(y_hat, y, cls))


# Test case 4
# Discrete Input and Real Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))
X,y=preprocessing(X,y)

for criteria in ["mse"]:
    tree = DecisionTree(criterion=criteria) 
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))



