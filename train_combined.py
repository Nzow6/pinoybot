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

# Specify the two files to combine and train on
FILE1 = "MCO2_Dataset_Cleaned.xlsx"
FILE2 = "annotated_dataset_updated.csv"

def map_labels(tag):
    if pd.isna(tag):
        return 'OTH'
    tag = str(tag).upper().strip()
    if 'FIL' in tag or 'CS' in tag:
        return 'FIL'
    if 'ENG' in tag:
        return 'ENG'
    return 'OTH'

# 1. Load File 1
print(f"Loading dataset 1: {FILE1}...")
if FILE1.endswith('.xlsx'):
    df1 = pd.read_excel(FILE1)
else:
    df1 = pd.read_csv(FILE1)
df1 = df1.dropna(subset=['word']).copy()
df1['final_tag'] = df1['corrected_label'].fillna(df1['label']) if 'corrected_label' in df1.columns else df1['label']
df1['mapped_label'] = df1['final_tag'].apply(map_labels)

# 2. Load File 2
print(f"Loading dataset 2: {FILE2}...")
if FILE2.endswith('.xlsx'):
    df2 = pd.read_excel(FILE2)
else:
    df2 = pd.read_csv(FILE2, keep_default_na=False)

df2['word'] = df2['word'].astype(str)
df2 = df2[df2['word'].str.strip() != ''].copy()

tag_col = 'tag' if 'tag' in df2.columns else ('corrected_label' if 'corrected_label' in df2.columns else 'label')
df2['mapped_label'] = df2[tag_col].apply(map_labels)

# Combine datasets
print(f"\nMerging datasets ({len(df1)} rows + {len(df2)} rows)...")
df_combined = pd.concat([df1[['word', 'mapped_label']], df2[['word', 'mapped_label']]], ignore_index=True)
print(f"Total combined rows: {len(df_combined)}")

print("\nCombined Label Distribution:")
print(df_combined['mapped_label'].value_counts())

feature_dicts = []
prev_word, prev_pred = None, None

print("\nExtracting features from combined datasets...")
start_feat = time.time()

for i, row in df_combined.iterrows():
    word_str = str(row['word'])
    feats = extract_features_for_word(word_str, prev_word, prev_pred)
    feature_dicts.append(feats)

    # Reset context at sentence boundaries
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
y = df_combined['mapped_label'].values
end_vec = time.time()

print(f"Vectorization time: {end_vec - start_vec:.2f} seconds")
print("Combined feature matrix shape:", X.shape)

# Stratified 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 1. RANDOM FOREST
print("\n" + "="*60)
print("1. TRAINING RANDOM FOREST CLASSIFIER ON COMBINED DATA")
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

# 2. SVM
print("\n" + "="*60)
print("2. TRAINING SUPPORT VECTOR MACHINE ON COMBINED DATA")
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

# Save models and vectorizer
joblib.dump(rfc, "pinoybot_model_random_forest.pkl")
joblib.dump(svm, "pinoybot_model_svm.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")

print("\nSaved combined trained models: pinoybot_model_random_forest.pkl, pinoybot_model_svm.pkl, and pinoybot_vectorizer.pkl")
