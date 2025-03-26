import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('faces_embeddings.csv')

X = df.iloc[:, :128]
y = df['gender']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_scores = []
test_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"\nFold {fold}:")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    if fold == 1:
        feature_importance = pd.DataFrame({
            'Feature': [f'X{i}' for i in range(128)],
            'Importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

print("\nAverage Training Accuracy:", np.mean(train_scores))
print("Average Testing Accuracy:", np.mean(test_scores))

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature')
plt.title('Top 20 Most Important Features for Gender Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

final_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
final_model.fit(X_scaled, y)

y_pred = final_model.predict(X_scaled)
print("\nFinal Model Performance:")
print(classification_report(y, y_pred)) 