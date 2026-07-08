
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\preprocessed_credit_data.csv")
print(" Data Loaded Successfully!")
print("Columns:", df.columns)


X=df.drop(['Credit amount','Credit risk'],axis=1)
y=df['Credit risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Data Split Completed!")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(" SMOTE Applied (Balanced Data)")

rf_model = RandomForestClassifier(
    n_estimators=100,      
    criterion='gini',      
    max_depth=8,           
    random_state=42,
    n_jobs=-1              
)

rf_model.fit(X_train, y_train)
print(" Model Trained Successfully!")

y_pred = rf_model.predict(X_test)

print("\n Model Evaluation:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))