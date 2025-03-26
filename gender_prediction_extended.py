import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')

df = pd.read_csv('faces_embeddings.csv')

X = df.iloc[:, :128]
y = df['gender']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def evaluate_model(C_value, X_scaled, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_scores = []
    test_scores = []
    feature_importances = None
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C_value, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        if fold == 1:
            feature_importances = abs(model.coef_[0])
    
    return np.mean(train_scores), np.mean(test_scores), feature_importances

C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
results = []

for C in C_values:
    train_score, test_score, importances = evaluate_model(C, X_scaled, y)
    results.append({
        'C': C,
        'train_score': train_score,
        'test_score': test_score,
        'feature_importances': importances,
        'num_nonzero_features': np.sum(importances > 0.001)
    })
    print(f"\nC = {C}:")
    print(f"Average Training Accuracy: {train_score:.4f}")
    print(f"Average Testing Accuracy: {test_score:.4f}")
    print(f"Number of important features: {np.sum(importances > 0.001)}")

plt.figure(figsize=(10, 6))
plt.plot([r['C'] for r in results], [r['train_score'] for r in results], 'o-', label='Training Accuracy')
plt.plot([r['C'] for r in results], [r['test_score'] for r in results], 'o-', label='Testing Accuracy')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Model Performance vs Regularization Strength')
plt.legend()
plt.grid(True)
plt.savefig('regularization_performance.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot([r['C'] for r in results], [r['num_nonzero_features'] for r in results], 'o-')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Number of Important Features')
plt.title('Feature Sparsity vs Regularization Strength')
plt.grid(True)
plt.savefig('feature_sparsity.png')
plt.close()

plt.figure(figsize=(15, 8))
feature_importance_data = []
for i, C in enumerate(C_values):
    importances = results[i]['feature_importances']
    top_features = np.argsort(importances)[-10:]
    for feat_idx in top_features:
        feature_importance_data.append({
            'C': C,
            'Feature': f'X{feat_idx}',
            'Importance': importances[feat_idx]
        })

importance_df = pd.DataFrame(feature_importance_data)
pivot_df = importance_df.pivot(index='Feature', columns='C', values='Importance')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Top Feature Importance Across Regularization Strengths')
plt.xlabel('Regularization Strength (C)')
plt.savefig('feature_importance_heatmap.png')
plt.close()

best_result = max(results, key=lambda x: x['test_score'])
print("\nOptimal Regularization Strength:")
print(f"C = {best_result['C']}")
print(f"Training Accuracy: {best_result['train_score']:.4f}")
print(f"Testing Accuracy: {best_result['test_score']:.4f}")

optimal_importances = pd.DataFrame({
    'Feature': [f'X{i}' for i in range(128)],
    'Importance': best_result['feature_importances']
})
optimal_importances = optimal_importances.sort_values('Importance', ascending=False)
print("\nTop 10 Most Important Features (Optimal Model):")
print(optimal_importances.head(10)) 