
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"E:\M.tech\New_Research\Dataset\preprocessed_credit_data.csv") 

print("Dataset Preview:\n", df.head())

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Credit risk', axis=1)    
y = df['Credit risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.title("Feature Importance using SHAP Values (Random Forest)")
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
plt.savefig("E:/M.tech/New_Research/shap_summary_bar.png", dpi=300, bbox_inches='tight')
plt.show()

shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig("E:/M.tech/New_Research/shap_summary_detailed.png", dpi=300, bbox_inches='tight')
plt.show()

sample_index = 5
shap.force_plot(
    explainer.expected_value[1], shap_values[1][sample_index, :], 
    X_test.iloc[sample_index, :], matplotlib=True
)
