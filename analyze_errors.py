import joblib
import pathlib
import sys
from collections import Counter
from typing import List
from feature_utils import extract_features_for_word

sys.stdout.reconfigure(encoding='utf-8')

script_dir = pathlib.Path(__file__).parent
DATA_PATH = script_dir / "test sentences" / "test" / "test_data.txt"
LABELS_PATH = script_dir / "test sentences" / "test" / "test_labels.txt"

VEC_PATH = script_dir / "pinoybot_vectorizer.pkl"
RFC_PATH = script_dir / "pinoybot_model_random_forest.pkl"

vec = joblib.load(VEC_PATH)
clf_rfc = joblib.load(RFC_PATH)

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
        X_tmp = vec.transform([feats])
        step_pred = clf.predict(X_tmp)[0]
        prev_word = word
        prev_pred = step_pred

    X_new = vec.transform(features)
    predicted = clf.predict(X_new)
    return [str(tag) for tag in predicted]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_lines = f.readlines()
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_lines = f.readlines()

SVM_PATH = script_dir / "pinoybot_model_svm.pkl"
clf_svm = joblib.load(SVM_PATH)

def run_error_analysis(clf_model, model_name):
    eng_false_positives = [] # Actual FIL/OTH predicted as ENG
    oth_false_negatives = [] # Actual OTH predicted as FIL/ENG

    for data_line, label_line in zip(data_lines, label_lines):
        tokens = [t.strip() for t in data_line.split("|") if t and t.strip()]
        labels = [l.strip() for l in label_line.split("|") if l and l.strip()]
        min_len = min(len(tokens), len(labels))
        tokens, labels = tokens[:min_len], labels[:min_len]

        preds = tag_language(tokens, clf_model)

        for tok, true_lbl, pred_lbl in zip(tokens, labels, preds):
            if true_lbl != "ENG" and pred_lbl == "ENG":
                eng_false_positives.append((tok, true_lbl))
            if true_lbl == "OTH" and pred_lbl != "OTH":
                oth_false_negatives.append((tok, pred_lbl))

    print("\n" + "="*60)
    print(f"ERROR ANALYSIS REPORT ({model_name})")
    print("="*60)

    print(f"Total English False Positives: {len(eng_false_positives)}")
    fp_counts = Counter([tok.lower() for tok, _ in eng_false_positives])
    print("Most Frequent English False Positive Tokens:")
    for tok, count in fp_counts.most_common(10):
        print(f"  - '{tok}': {count} instances")

    homographs = {"dating", "in", "man", "tan", "on", "so", "am", "do", "go", "me", "he", "we", "is", "it", "or", "to", "be", "no", "at"}
    homograph_fp_count = sum(count for tok, count in fp_counts.items() if tok in homographs)
    pct_homograph = (homograph_fp_count / len(eng_false_positives)) * 100 if eng_false_positives else 0

    print(f"Cross-Language Homograph False Positives: {homograph_fp_count} / {len(eng_false_positives)} ({pct_homograph:.1f}%)")

    print(f"\nTotal Missed OTH Tokens (OTH False Negatives): {len(oth_false_negatives)}")
    oth_fn_counts = Counter([tok for tok, _ in oth_false_negatives])
    print("Most Frequent Missed OTH Tokens:")
    for tok, count in oth_fn_counts.most_common(15):
        print(f"  - '{tok}': {count} instances")

run_error_analysis(clf_rfc, "RANDOM FOREST CLASSIFIER")
run_error_analysis(clf_svm, "SUPPORT VECTOR MACHINE")
