import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('faces_embeddings.csv')

# Separate features and target
# Features are columns 0-127, target is 'gender'
X = df.iloc[:, :128]  # First 128 columns are the features
y = df['gender']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize k-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
train_scores = []
test_scores = []

# Perform k-fold cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train logistic regression with L1 regularization (Lasso)
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate scores
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"\nFold {fold}:")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Print feature importance for the first fold
    if fold == 1:
        feature_importance = pd.DataFrame({
            'Feature': [f'X{i}' for i in range(128)],
            'Importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

# Print average scores
print("\nAverage Training Accuracy:", np.mean(train_scores))
print("Average Testing Accuracy:", np.mean(test_scores))

# Plot feature importance for the first fold
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature')
plt.title('Top 20 Most Important Features for Gender Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Train final model on all data
final_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
final_model.fit(X_scaled, y)

# Print final model performance
y_pred = final_model.predict(X_scaled)
print("\nFinal Model Performance:")
print(classification_report(y, y_pred)) 