import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from feature_utils import extract_features_for_word   
import joblib


df = pd.read_excel("MCO2 Dataset (full).xlsx")
df.head(10)
df = df.dropna(subset=['word']).copy()


# i placed this since corrected_label has null values since di pa navavalidate lahat ng labels

df['final_label'] = df['corrected_label'].fillna(df['label'])
df.head(10)
df['label'].value_counts()
# i just checked why there are NAN values in the word column

df['word'].isna().sum()

# why tf are there NAN values in the word column
# this should be labeled as dirty siguro in the first place ill just drop em
df[df['word'].isna()]

# since we only have 3 labels, we map all subcategories to main categories ('FIL', 'ENG', 'OTH')
def map_labels(tag):
    
    if pd.isna(tag):
        return 'OTH'
    tag = tag.upper()
    if tag == 'FIL'or tag == 'CS':
        return 'FIL'
    if tag == 'ENG':
        return 'ENG'
    
    return 'OTH'

df['mapped_label'] = df['final_label'].apply(map_labels)
df['mapped_label'].value_counts()


# this part will be for feature engineering and our model training
# prompt engineering at its finest. we can always add features kasi medj sinabi ko lang sakanya kung ano mga pwedeng icheck sa word

# we now use the extract_features_for_word() from feature_utils.py
feature_dicts = []
prev_word, prev_pred = None, None

for i, row in df.iterrows():
    feats = extract_features_for_word(row['word'], prev_word)
    feature_dicts.append(feats)
    
    prev_word = row['word']
    prev_pred = row['mapped_label']

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(feature_dicts)
y = df['mapped_label'].values


# testing 70% of the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

#15/15 split
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# trying random forest classifier instead of decision tree
rfc = RandomForestClassifier(
    n_estimators=400,         
    max_depth=18,              
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',      
    class_weight='balanced',   
    criterion='gini',         
    n_jobs=-1,
    random_state=42
)

rfc.fit(X_train, y_train)


rfc.fit(X_train, y_train)
print("Training Accuracy:", rfc.score(X_train, y_train))    
print("Validation Accuracy:", rfc.score(X_val, y_val))
print("Test Accuracy:", rfc.score(X_test, y_test))

# just a simple report

y_val_pred = rfc.predict(X_val)
y_test_pred = rfc.predict(X_test)


print("Validation Report:")
print(classification_report(y_val, y_val_pred))

print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nTest Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))



joblib.dump(rfc, "pinoybot_model.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")
print("\nModel and vectorizer saved successfully.")

