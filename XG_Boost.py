
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\preprocessed_credit_data.csv")
print(" Data Loaded Successfully!")
print("Columns:", df.columns)

X=df.drop(['Credit amount','Credit risk'],axis=1)
y=df['Credit risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Train-Test Split Completed!")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(" SMOTE Applied (Balanced Data)")

xgb_model = XGBClassifier(
    n_estimators=200,         
    learning_rate=0.1,        
    max_depth=6,              
    subsample=0.8,            
    colsample_bytree=0.8,     
    random_state=42,
    eval_metric='logloss',    
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
print(" Model Trained Successfully!")

y_pred = xgb_model.predict(X_test)

print("\n Model Evaluation:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = xgb_model.feature_importances_
sns.barplot(x=feature_importance, y=X.columns)
plt.title("Feature Importance (XGBoost)")
plt.show()
