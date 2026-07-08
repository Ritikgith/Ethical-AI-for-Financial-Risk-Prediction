import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r"E:\M.tech\New_Research\Dataset\german_credit_data.csv")
print(df.head(5))
print(df['Sex'].isnull().sum())

print(df.isnull().sum())
print(df.isnull())
print(df.head(20).isnull())

print(df.shape)

df.fillna(df.median(numeric_only=True), inplace=True)
print("Missing Values Handled!")


cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical Columns Encoded!")
print(df.head(20))

scale_cols = ['Age', 'Credit amount', 'Duration']
sc = StandardScaler()
df[scale_cols] = sc.fit_transform(df[scale_cols])
print("Feature Scaling Completed!")
print(df)

print(df.shape)

median_credit = df['Credit amount'].median()
df['Credit risk'] = df['Credit amount'].apply(lambda x: 1 if x > median_credit else 0)
print("Target Column 'Credit risk' Created!")

print(df)

df.to_csv("E:/M.tech/New_Research/Dataset/preprocessed_credit_data.csv", index=False)
print("Preprocessed file saved successfully!")
