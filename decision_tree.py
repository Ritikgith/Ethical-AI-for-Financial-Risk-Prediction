
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\preprocessed_credit_data.csv")
print("Data Loaded Successfully!")
print("Columns:", df.columns)

X=df.drop(['Credit amount','Credit risk'],axis=1)
y=df['Credit risk']             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split Completed!")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("SMOTE Applied (Balanced Data)")

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

dt_model.fit(X_train, y_train)
print("Model Trained Successfully!")

y_pred = dt_model.predict(X_test)

print("\nModel Evaluation:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))