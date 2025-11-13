"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""
import joblib
from typing import List
from feature_utils import extract_features_for_word
import pathlib
import re
import os
#simple sanity check to know where the current working directory is (sometimes different from the directory of the file itself)
print(f"I am looking for the file in this directory: {os.getcwd()}")

# Main tagging function
"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier
Loads the trained Decision Tree model and vectorizer to tag new tokens.
"""

# Load model + vectorizer
MODEL_PATH = "pinoybot_model.pkl"
VEC_PATH = "pinoybot_vectorizer.pkl"

# alternative for file loading using absolute paths 
script_dir = pathlib.Path(__file__).parent 

file_path = script_dir / MODEL_PATH
clf = joblib.load(file_path)

file_path = script_dir / VEC_PATH
vec = joblib.load(file_path)


# clf = joblib.load(MODEL_PATH)
# vec = joblib.load(VEC_PATH)

def decade_to_word(decade):
    decade = decade.lower()
    if (len(decade) <3):
        return decade # early return to avoid out of range errors
    if decade[-1] == 's' and decade[-2].isdigit() or decade[-2] == "'" and decade[-1] == 's' and decade[-3].isdigit():
        decade_str = decade.lower()
        decade_str = decade.replace("'", "")  # remove apostrophe
        decade_str = decade_str.replace('s', '')

        if not decade_str or not decade_str.isdigit():
            return decade

        year_int = int(decade_str)
        decade_num = year_int % 100
        
        # Map numbers to their word equivalents
        number_words = {
            0: 'hundreds', 10: 'tens', 20: 'twenties', 30: 'thirties',
            40: 'forties', 50: 'fifties', 60: 'sixties', 70: 'seventies',
            80: 'eighties', 90: 'nineties'
        }
        
        return number_words.get(decade_num, decade)
    else:
        return decade

def tag_language(tokens: List[str]) -> List[str]:
    """Tag each token as FIL, ENG, or OTH."""
    token_copy = [decade_to_word(word) for word in tokens]
    features = [extract_features_for_word(word) for word in token_copy]
    X_new = vec.transform(features)
    predicted = clf.predict(X_new)
    return [str(tag) for tag in predicted]

if __name__ == "__main__":
    
    sentence = "I heard you started dating her ah! Kamusta naman yung dating niya sayo? "

    punctuation_to_separate = r'([.,;:\"?!()])'
    tokens = re.split(r'\s+|' + punctuation_to_separate, sentence)
    tokens = [token for token in tokens if token and token.strip()]

    example_tokens=tokens



    predicted_tags = tag_language(example_tokens) 

    print("TAG | TOKEN")
    for i in range(len(example_tokens)):
        token = example_tokens[i]
        tag = predicted_tags[i]
        print(f"{tag} | {token}")

    # print("Tokens:", example_tokens)
    # print("Predicted tags:", tag_language(example_tokens))
