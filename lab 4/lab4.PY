
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ========================================
# BAI 1: XAY DUNG MO HINH LOGISTIC REGRESSION
# ========================================

print("========== BAI 1 ==========")

# Load du lieu breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

print("So luong mau:", X.shape[0])
print("So luong dac trung:", X.shape[1])

# Chia du lieu Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuan hoa du lieu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huan luyen mo hinh
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Huan luyen mo hinh thanh cong!")


# ========================================
# BAI 2: DANH GIA MO HINH PHAN LOAI
# ========================================

print("\n========== BAI 2 ==========")

# Du doan tren tap Test
y_pred = model.predict(X_test)

# Tinh cac chi so danh gia
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Bao cao phan loai chi tiet
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ========================================
# BAI 3: CONFUSION MATRIX
# ========================================

print("\n========== BAI 3 ==========")

# Tao ma tran nham lan
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Giai thich:
# TN: Du doan 0, thuc te 0
# FP: Du doan 1, thuc te 0
# FN: Du doan 0, thuc te 1
# TP: Du doan 1, thuc te 1

tn, fp, fn, tp = cm.ravel()

print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

# Ve Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Breast Cancer")
plt.show()


# ========================================
# BAI 4: LOGISTIC REGRESSION VOI DIGITS
# ========================================

print("\n========== BAI 4 ==========")

# Load du lieu digits
digits = load_digits()

X = digits.data
y = digits.target

print("So luong mau:", X.shape[0])
print("So luong dac trung:", X.shape[1])

# Chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuan hoa du lieu
scaler_digits = StandardScaler()

X_train = scaler_digits.fit_transform(X_train)
X_test = scaler_digits.transform(X_test)

# Huan luyen Logistic Regression
model_digits = LogisticRegression(max_iter=2000)
model_digits.fit(X_train, y_train)

# Du doan
y_pred_digits = model_digits.predict(X_test)

# Danh gia mo hinh
print("\nAccuracy:", accuracy_score(y_test, y_pred_digits))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_digits))

# Tim cac mau du doan sai
wrong_indices = np.where(y_test != y_pred_digits)[0]

print("\n5 mau du doan sai:")

for i in wrong_indices[:5]:
    print(
        "Nhan thuc te:", y_test[i],
        "- Nhan du doan:", y_pred_digits[i]
    )

# Hien thi 5 anh du doan sai
if len(wrong_indices) > 0:
    n = min(5, len(wrong_indices))
    fig, axes = plt.subplots(1, n, figsize=(12, 3))

    axes = np.atleast_1d(axes)

    for ax, i in zip(axes, wrong_indices[:n]):
        ax.imshow(
            X_test[i].reshape(8, 8),
            cmap="gray"
        )
        ax.set_title(
            f"True: {y_test[i]}\nPred: {y_pred_digits[i]}"
        )
        ax.axis("off")

    plt.suptitle("5 Mau Du Doan Sai - Digits")
    plt.tight_layout()
    plt.show()

print("\nHOAN THANH LAB 4!")
