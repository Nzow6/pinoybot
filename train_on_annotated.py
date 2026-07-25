import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from feature_utils import extract_features_for_word
import joblib
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

DATASET_PATH = "annotated_dataset_updated.csv"

df = pd.read_csv(DATASET_PATH, keep_default_na=False)
df['word'] = df['word'].astype(str)
df = df[df['word'].str.strip() != ''].copy()
print(f"Total rows loaded from {DATASET_PATH}: {df.shape[0]}")

def map_labels(tag):
    if pd.isna(tag) or tag == '':
        return 'OTH'
    tag = str(tag).upper().strip()
    if 'FIL' in tag or 'CS' in tag:
        return 'FIL'
    if 'ENG' in tag:
        return 'ENG'
    return 'OTH'

df['mapped_label'] = df['tag'].apply(map_labels)
print("\nMapped label counts:")
print(df['mapped_label'].value_counts())

feature_dicts = []
prev_word, prev_pred = None, None

print("\nExtracting features...")
start_feat = time.time()

for i, row in df.iterrows():
    word_str = str(row['word'])
    feats = extract_features_for_word(word_str, prev_word, prev_pred)
    feature_dicts.append(feats)

    # Reset context at sentence boundary punctuation
    if word_str in ['.', '!', '?']:
        prev_word, prev_pred = None, None
    else:
        prev_word = word_str
        prev_pred = row['mapped_label']

end_feat = time.time()
print(f"Feature extraction time: {end_feat - start_feat:.2f} seconds")

vec = DictVectorizer(sparse=True)
start_vec = time.time()
X = vec.fit_transform(feature_dicts)
y = df['mapped_label'].values
end_vec = time.time()

print(f"Vectorization time: {end_vec - start_vec:.2f} seconds")
print("Feature matrix shape:", X.shape)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 1. RANDOM FOREST
print("\n" + "="*60)
print("1. TRAINING RANDOM FOREST CLASSIFIER")
print("="*60)

rfc = RandomForestClassifier(
    n_estimators=120,
    max_depth=80,
    max_features="log2",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
start_rf = time.time()
rfc.fit(X_train, y_train)
end_rf = time.time()
print(f"Random Forest Training time: {end_rf - start_rf:.2f} seconds")

print(f"RFC Train Accuracy: {rfc.score(X_train, y_train):.4f}")
print(f"RFC Val Accuracy:   {rfc.score(X_val, y_val):.4f}")
print(f"RFC Test Accuracy:  {rfc.score(X_test, y_test):.4f}")

print("\n=== RFC Classification Report (Validation) ===")
print(classification_report(y_val, rfc.predict(X_val), digits=4))

print("\n=== RFC Confusion Matrix (Validation) ===")
print(confusion_matrix(y_val, rfc.predict(X_val), labels=rfc.classes_))

# 2. SVM
print("\n" + "="*60)
print("2. TRAINING SUPPORT VECTOR MACHINE (LinearSVC)")
print("="*60)

svm = LinearSVC(
    class_weight='balanced',
    random_state=42,
    max_iter=5000,
    dual="auto"
)
start_svm = time.time()
svm.fit(X_train, y_train)
end_svm = time.time()
print(f"SVM Training time: {end_svm - start_svm:.2f} seconds")

print(f"SVM Train Accuracy: {svm.score(X_train, y_train):.4f}")
print(f"SVM Val Accuracy:   {svm.score(X_val, y_val):.4f}")
print(f"SVM Test Accuracy:  {svm.score(X_test, y_test):.4f}")

print("\n=== SVM Classification Report (Validation) ===")
print(classification_report(y_val, svm.predict(X_val), digits=4))

print("\n=== SVM Confusion Matrix (Validation) ===")
print(confusion_matrix(y_val, svm.predict(X_val), labels=svm.classes_))

# Save models and vectorizer
joblib.dump(rfc, "pinoybot_model_random_forest.pkl")
joblib.dump(svm, "pinoybot_model_svm.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")

print("\nSaved pinoybot_model_random_forest.pkl, pinoybot_model_svm.pkl, and pinoybot_vectorizer.pkl successfully.")
