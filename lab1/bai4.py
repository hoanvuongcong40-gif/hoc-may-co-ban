
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

print("Du lieu:", digits.data.shape)
print("Nhan:", digits.target.shape)

plt.imshow(digits.images[0], cmap="gray")
plt.title("So: " + str(digits.target[0]))
plt.show()
