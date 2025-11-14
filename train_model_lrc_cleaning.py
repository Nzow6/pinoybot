import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from feature_utils import extract_features_for_word
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')


DEV_MODE = False  # Set to True if you want only CV
DATASET_PATH = "MCO2 Dataset (full).xlsx"


df = pd.read_excel(DATASET_PATH)
print(f"Total raw rows loaded: {df.shape[0]}")

df = df.dropna(subset=['word']).copy()



def normalize_is_correct(x):
    """Convert is_correct column to clean True/False/NA."""
    if pd.isna(x):
        return pd.NA
    if isinstance(x, bool):
        return x
    s = str(x).strip().upper()
    if s == "TRUE":
        return True
    if s == "FALSE":
        return False
    return pd.NA

# Normalize
df["is_correct_norm"] = df["is_correct"].apply(normalize_is_correct)

# Priority ranking: True (0) → False (1) → NaN (2)
priority_map = {True: 0, False: 1}
df["priority"] = df["is_correct_norm"].map(priority_map).fillna(2).astype(int)

# Sort so the “best” row per token appears first
df = df.sort_values(
    by=["sentence_id", "word_id", "priority"],
    ascending=[True, True, True]
)


df = df.drop_duplicates(
    subset=["sentence_id", "word_id"],
    keep="first"
).copy()

print(f"Rows remaining after cleaning: {df.shape[0]}")


df['final_label'] = df['corrected_label'].fillna(df['label'])


def map_labels(tag):
    if pd.isna(tag):
        return 'OTH'
    tag = str(tag).upper()
    
    if 'FIL' in tag or 'CS' in tag:
        return 'FIL'
    if 'ENG' in tag:
        return 'ENG'
    
    return 'OTH'

df['mapped_label'] = df['final_label'].apply(map_labels)

feature_dicts = []
prev_word, prev_pred = None, None

for i, row in df.iterrows():
    feats = extract_features_for_word(row['word'], prev_word)
    feature_dicts.append(feats)
    prev_word = row['word']
    prev_pred = row['mapped_label']

# Vectorize features
vec = DictVectorizer(sparse=True)
X = vec.fit_transform(feature_dicts)
y = df['mapped_label'].values


log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# ------------------------
# DEV MODE (cross validation checkerk)
# ------------------------
if DEV_MODE:
    print("\n=== DEVELOPMENT MODE: Running 5-Fold Cross Validation ===")
    cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print("Cross-Validation Accuracies:", np.round(cv_scores, 4))
    print(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nSkipping model saving in DEV mode.")
    sys.exit()


print("\n=== Training Final Model ===")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

log_reg.fit(X_train, y_train)

train_acc = log_reg.score(X_train, y_train)
val_acc = log_reg.score(X_val, y_val)
test_acc = log_reg.score(X_test, y_test)

print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

# Reports
print("\nClassification Report (Validation):")
print(classification_report(y_val, log_reg.predict(X_val), digits=3))

print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, log_reg.predict(X_val), labels=log_reg.classes_))
print("-" * 60)


print("\n=== MODEL PERFORMANCE SUMMARY ===")
print(f"{'Model':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print("-" * 44)
print(f"{'Logistic Regression':20s} {train_acc:8.4f} {val_acc:8.4f} {test_acc:8.4f}")


print("\n=== TOP 10 FEATURES PER CLASS ===")
feature_names = vec.get_feature_names_out()
coef = log_reg.coef_
classes = log_reg.classes_

for class_idx, class_name in enumerate(classes):
    print(f"\nTop features for class: {class_name}")
    top_indices = np.argsort(np.abs(coef[class_idx]))[::-1][:10]
    for i in top_indices:
        weight = coef[class_idx][i]
        direction = "increases likelihood" if weight > 0 else "decreases likelihood"
        print(f"{feature_names[i]:30s} {weight:+.3f}  {direction}")
    print("-" * 60)

# Save model
joblib.dump(log_reg, "pinoybot_model.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")
print("Model and vectorizer saved successfully!")
