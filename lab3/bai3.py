
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes()
X = data.data[:, 2:3]
y = data.target

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, label="Thuc te")
plt.plot(X, model.predict(X), color="red", label="Du doan")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
