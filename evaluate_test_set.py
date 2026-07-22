import joblib
import pathlib
import sys
from typing import List
from sklearn.metrics import classification_report, confusion_matrix
from feature_utils import extract_features_for_word

sys.stdout.reconfigure(encoding='utf-8')

# Paths
script_dir = pathlib.Path(__file__).parent
DATA_PATH = script_dir / "test sentences" / "test" / "test_data.txt"
LABELS_PATH = script_dir / "test sentences" / "test" / "test_labels.txt"

VEC_PATH = script_dir / "pinoybot_vectorizer.pkl"
RFC_PATH = script_dir / "pinoybot_model_random_forest.pkl"
SVM_PATH = script_dir / "pinoybot_model_svm.pkl"

# Load common vectorizer
print("Loading vectorizer...")
vec = joblib.load(VEC_PATH)

# Load models
print("Loading models...")
clf_rfc = joblib.load(RFC_PATH)
clf_svm = joblib.load(SVM_PATH)

def decade_to_word(decade):
    decade = decade.lower()
    if len(decade) < 3:
        return decade
    if (decade[-1] == 's' and decade[-2].isdigit()) or (decade[-2] == "'" and decade[-1] == 's' and decade[-3].isdigit()):
        decade_str = decade.replace("'", "").replace('s', '')
        if not decade_str or not decade_str.isdigit():
            return decade
        year_int = int(decade_str)
        decade_num = year_int % 100
        number_words = {
            0: 'hundreds', 10: 'tens', 20: 'twenties', 30: 'thirties',
            40: 'forties', 50: 'fifties', 60: 'sixties', 70: 'seventies',
            80: 'eighties', 90: 'nineties'
        }
        return number_words.get(decade_num, decade)
    return decade

def tag_language(tokens: List[str], clf) -> List[str]:
    token_copy = [decade_to_word(word) for word in tokens]
    features = []
    prev_word = None
    prev_pred = None

    for word in token_copy:
        feats = extract_features_for_word(word, prev_word, prev_pred)
        features.append(feats)

        # Temporary prediction for contextual feature flow
        X_tmp = vec.transform([feats])
        step_pred = clf.predict(X_tmp)[0]
        prev_word = word
        prev_pred = step_pred

    X_new = vec.transform(features)
    predicted = clf.predict(X_new)
    return [str(tag) for tag in predicted]

# Read external test dataset
print(f"Reading test data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_lines = f.readlines()

print(f"Reading test labels from: {LABELS_PATH}")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_lines = f.readlines()

# Verify matching line counts
if len(data_lines) != len(label_lines):
    print(f"Warning: Line counts mismatch! Data: {len(data_lines)}, Labels: {len(label_lines)}")

y_true = []
y_pred_rfc = []
y_pred_svm = []

mismatched_tokens_count = 0

for idx, (data_line, label_line) in enumerate(zip(data_lines, label_lines)):
    tokens = [t.strip() for t in data_line.split("|") if t and t.strip()]
    labels = [l.strip() for l in label_line.split("|") if l and l.strip()]

    if len(tokens) != len(labels):
        mismatched_tokens_count += 1
        # Sync to minimum length to prevent index errors
        min_len = min(len(tokens), len(labels))
        tokens = tokens[:min_len]
        labels = labels[:min_len]

    # Predict tags
    pred_rfc = tag_language(tokens, clf_rfc)
    pred_svm = tag_language(tokens, clf_svm)

    y_true.extend(labels)
    y_pred_rfc.extend(pred_rfc)
    y_pred_svm.extend(pred_svm)

print(f"Processed {len(data_lines)} sentences. Found {mismatched_tokens_count} sentences with token-label mismatches.")
print(f"Total tokens evaluated: {len(y_true)}")

# ============================================================
# RESULTS REPORTING
# ============================================================
print("\n" + "="*60)
print("EVALUATION ON PROFESSOR'S TEST SENTENCES")
print("="*60)

print("\n=== RANDOM FOREST CLASSIFIER (RFC) ===")
print("Classification Report:")
print(classification_report(y_true, y_pred_rfc, digits=4))
print("Confusion Matrix (FIL, ENG, OTH):")
print(confusion_matrix(y_true, y_pred_rfc, labels=["FIL", "ENG", "OTH"]))

print("\n" + "-"*60)

print("\n=== SUPPORT VECTOR MACHINE (SVM) ===")
print("Classification Report:")
print(classification_report(y_true, y_pred_svm, digits=4))
print("Confusion Matrix (FIL, ENG, OTH):")
print(confusion_matrix(y_true, y_pred_svm, labels=["FIL", "ENG", "OTH"]))
