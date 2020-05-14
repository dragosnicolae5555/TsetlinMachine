from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df1 = pd.read_csv('QD_BIN_FINAL.csv', header = None)
df2 = pd.read_csv('label.csv', header = None)
x = df1.iloc[:,:].values
y = df2.iloc[:,:].values
y = np.reshape(y, len(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.084, shuffle=False)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print('data fitting started..........................')


clf = [DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=3),GaussianNB(),
LogisticRegression(random_state=0),MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),
       svm.LinearSVC()]
for algo in clf:
    algo.fit(x_train, y_train)

    y1 = algo.predict(x_train)
    y_pred1 = np.array(y1)
    from sklearn import metrics

    acc1 = metrics.accuracy_score(y_train, y_pred1)
    print('training ',acc1)

    y2 = algo.predict(x_test)
    y_pred2 = np.array(y2)
    acc2 = metrics.accuracy_score(y_test, y_pred2)
    print('Testing ',acc2)