import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from feature_utils import extract_features_for_word   
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Read dataset and clean up
df = pd.read_excel("MCO2 Dataset (full).xlsx")
df = df.dropna(subset=['word']).copy()

# Handle missing corrected labels
df['final_label'] = df['corrected_label'].fillna(df['label'])

# Map labels to three categories: FIL, ENG, OTH
def map_labels(tag):
    if pd.isna(tag):
        return 'OTH'
    tag = tag.upper()
    if tag == 'FIL' or tag == 'CS':
        return 'FIL'
    if tag == 'ENG':
        return 'ENG'
    return 'OTH'

df['mapped_label'] = df['final_label'].apply(map_labels)

# Feature extraction from words
feature_dicts = []
prev_word, prev_pred = None, None
for i, row in df.iterrows():
    feats = extract_features_for_word(row['word'], prev_word)
    feature_dicts.append(feats)
    prev_word = row['word']
    prev_pred = row['mapped_label']

# Vectorizing the features
vec = DictVectorizer(sparse=True)
X = vec.fit_transform(feature_dicts)
y = df['mapped_label'].values

# Splitting data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Logistic Regression model
print("\n=== Logistic Regression ===")

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# Evaluate the model
train_acc = log_reg.score(X_train, y_train)
val_acc = log_reg.score(X_val, y_val)
test_acc = log_reg.score(X_test, y_test)

print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

# Print Classification Report
print("\nClassification Report (Validation):")
print(classification_report(y_val, log_reg.predict(X_val), digits=3))

# Confusion Matrix
print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, log_reg.predict(X_val)))
print("-" * 60)

# Summary Table
print("\n=== MODEL PERFORMANCE SUMMARY ===")
print(f"{'Model':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print("-" * 44)
print(f"{'Logistic Regression':20s} {train_acc:8.4f} {val_acc:8.4f} {test_acc:8.4f}")

# Top 5 Most Significant Features for each class - Summary
print("\n=== TOP 5 FEATURES ===")

feature_names = vec.get_feature_names_out()
coefs = log_reg.coef_
classes = log_reg.classes_

for class_idx, class_name in enumerate(classes):
    print(f"\nTop features for class: {class_name}")
    top_indices = np.argsort(np.abs(coefs[class_idx]))[::-1][:5]
    for i in top_indices:
        weight = coefs[class_idx][i]
        direction = "increases likelihood" if weight > 0 else "decreases likelihood"
        print(f"{feature_names[i]:30s} {weight:+.3f}  {direction}")
    print("-" * 60)

# Save the model and vectorizer for deployment
joblib.dump(log_reg, "pinoybot_model.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")
print("Model and vectorizer saved successfully as pinoybot_model.pkl and pinoybot_vectorizer.pkl")
