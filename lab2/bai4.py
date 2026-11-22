
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

data = load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC())
])

model.fit(X_train, y_train)

print("Do chinh xac:", model.score(X_test, y_test))
