
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\preprocessed_credit_data.csv")

# X = df[['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
#         'Checking account', 'Credit amount','Duration', 'Purpose']]
# y = df['Credit risk'] 

X=df.drop(['Credit amount','Credit risk'],axis=1)
y=df['Credit risk']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("SMOTE Applied (Balanced Data)")

model = LogisticRegression()
model.fit(X_train, y_train)
print("Model Training Completed!")

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

