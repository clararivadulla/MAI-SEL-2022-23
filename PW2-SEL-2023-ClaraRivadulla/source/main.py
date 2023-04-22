import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_forest import DF
np.random.seed(2012)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
decision_forest = DF(max_depth=3, NT=10, F=4)
decision_forest.fit(X_train, y_train)
y_pred = decision_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
decision_forest = DF(max_depth=3, NT=10, F=4)
decision_forest.fit(X_train, y_train)
y_pred = decision_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
decision_forest = DF(max_depth=3, NT=10, F=4)
decision_forest.fit(X_train, y_train)
y_pred = decision_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

X, y = load_boston(return_X_y=True) # This doesn't work well
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
decision_forest = DF(max_depth=3, NT=2, F=10)
decision_forest.fit(X_train, y_train)
y_pred = decision_forest.predict(X_test)
print(y_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)