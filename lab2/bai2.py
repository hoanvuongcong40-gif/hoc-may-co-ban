
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.array([
    [20, 10],
    [30, 30],
    [40, 60],
    [60, 100]
])

print("Du lieu goc:", data)
print("StandardScaler:", StandardScaler().fit_transform(data))
print("MinMaxScaler:", MinMaxScaler().fit_transform(data))
