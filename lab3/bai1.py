
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.arange(1, 11).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.randn(10)

model = LinearRegression()
model.fit(X, y)

print("He so:", model.coef_)
print("Intercept:", model.intercept_)
