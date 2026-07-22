import joblib
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
print("Loading vectorizer...")
vec = joblib.load("pinoybot_vectorizer.pkl")

print("Loading Random Forest model...")
clf = joblib.load("pinoybot_model_random_forest.pkl")

feature_names = vec.get_feature_names_out()
importances = clf.feature_importances_

# Sort by importance (highest first)
sorted_indices = np.argsort(importances)[::-1]

print("\n" + "="*60)
print("TOP 20 MOST IMPORTANT FEATURES IN RANDOM FOREST (RFC)")
print("="*60)

print(f"{'Feature Name':40s} | {'Importance':>10s}")
print("-"*55)
for i in sorted_indices[:20]:
    score = importances[i]
    print(f"{feature_names[i]:40s} | {score:.6f}")
print("-"*55)
