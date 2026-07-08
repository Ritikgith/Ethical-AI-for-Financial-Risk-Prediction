import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\german_credit_data.csv")

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
sns.histplot(df['Age'], kde=True, color='skyblue', bins=25)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Density')

plt.subplot(2, 2, 2)
sns.histplot(df['Credit amount'], kde=True, color='salmon', bins=25)
plt.title('Distribution of Credit amount')
plt.xlabel('Credit amount')
plt.ylabel('Density')

plt.subplot(2, 2, 3)
sns.histplot(df['Duration'], kde=True, color='limegreen', bins=25)
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Density')

plt.subplot(2, 2, 4)
sns.countplot(x='Job', data=df, palette='viridis')
plt.title('Distribution of Job')
plt.xlabel('Job Type')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
