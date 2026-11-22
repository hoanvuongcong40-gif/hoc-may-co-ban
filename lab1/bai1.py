
import numpy as np
import pandas as pd

a = np.arange(1, 11)
print("Mang:", a)

df = pd.DataFrame({
    "Ten": ["An", "Binh", "Cuong", "Dung", "Hung"],
    "Tuoi": [18, 19, 20, 18, 19],
    "Diem": [8, 7, 9, 6, 10]
})

print(df)
df.info()
print("Diem trung binh:", df["Diem"].mean())
