
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

iris = load_iris()

X = iris.data
y = iris.target

model = LinearRegression()
model.fit(X, y)

print("Ket qua du doan:", model.predict(X[:5]))
