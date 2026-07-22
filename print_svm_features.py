import joblib
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
print("Loading vectorizer...")
vec = joblib.load("pinoybot_vectorizer.pkl")

print("Loading SVM model...")
clf = joblib.load("pinoybot_model_svm.pkl")

feature_names = vec.get_feature_names_out()
coefs = clf.coef_  # Shape: (n_classes, n_features)
classes = clf.classes_

print("\n" + "="*60)
print("TOP 10 FEATURES FOR EACH CLASS IN SVM")
print("="*60)

for class_idx, class_name in enumerate(classes):
    print(f"\n>>> Class: {class_name} <<<")
    # Sort features by coefficient value (highest positive weights first)
    sorted_indices = np.argsort(coefs[class_idx])[::-1]
    
    print(f"{'Feature Name':40s} | {'Weight':>8s}")
    print("-"*53)
    for i in sorted_indices[:10]:
        weight = coefs[class_idx][i]
        print(f"{feature_names[i]:40s} | {weight:+.4f}")
    print("-"*53)
