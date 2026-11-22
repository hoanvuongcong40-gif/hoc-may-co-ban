
import pandas as pd

df = pd.DataFrame({
    "Tuoi": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, None],
    "ThuNhap": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 1000]
})

df = df.fillna(df.mean())

tb = df["ThuNhap"].mean()
std = df["ThuNhap"].std()

loi = abs(df["ThuNhap"] - tb) > 3 * std
df.loc[loi, "ThuNhap"] = df.loc[~loi, "ThuNhap"].median()

print(df)
