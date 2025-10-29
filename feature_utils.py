# feature_utils.py
# -----------------------------------------------------------
# Ultimate version of the PinoyBot feature extractor
# Adds expanded Filipino & English morphology, 
# capitalization logic, short-word disambiguation, 
# and robust context features.
# -----------------------------------------------------------

import pandas as pd
import string
import re

VOWELS = set("aeiouAEIOU")

# English / Filipino pattern sets
ENGLISH_SUFFIXES = ("ing", "ed", "tion", "sion", "ly", "ment", "ness", "able", "ous", "ive", "less", "ful")
ENGLISH_CLUSTERS = ("th", "sh", "ch", "ph", "wh")
FILIPINO_PREFIXES = ("mag", "nag", "ni", "ka", "pa", "pag", "ma", "man", "na", "pin")
FILIPINO_SUFFIXES = ("an", "han", "hin", "in", "on", "ko", "ta", "ka", "mo")

COMMON_EN_SHORTS = {"i", "a", "an", "am", "is", "in", "it", "of", "on", "at", "to", "we", "he", "be", "do", "go", "no", "so", "up", "us", "the", "and"}
COMMON_FIL_SHORTS = {"si", "sa", "na", "pa", "po", "ko", "mo", "ka", "ba", "ha", "eh", "ay", "di"}

COMMON_ENG_WORDS = {"happy", "sad", "love", "study", "project", "graduation", "after", "before", "and", "today", "soon"}
COMMON_FIL_WORDS = {"grabe", "sige", "kasi", "naman", "ba", "po", "tapos", "ayan", "ako", "kami", "kayo", "pa", "na"}

def extract_features_for_word(word: str, prev_word=None, prev_pred=None):
    """Extract rich linguistic, orthographic, and contextual features for one word."""

    if not isinstance(word, str):
        word = "" if pd.isna(word) else str(word)
    lower = word.lower().strip()
    num_chars = len(lower)
    num_vowels = sum(ch in VOWELS for ch in lower)
    num_consonants = sum(ch.isalpha() and ch.lower() not in "aeiou" for ch in lower)
    vowel_ratio = (num_vowels / num_chars) if num_chars > 0 else 0.0
    consonant_ratio = (num_consonants / num_chars) if num_chars > 0 else 0.0

    feats = {
        # --- BASIC LEXICAL ---
        "length": num_chars,
        "num_vowels": num_vowels,
        "vowel_ratio": vowel_ratio,
        "consonant_ratio": consonant_ratio,
        "is_capitalized": int(num_chars > 0 and word[0].isupper()),
        "is_all_caps": int(num_chars > 0 and word.isupper()),
        "has_digit": int(any(ch.isdigit() for ch in word)),
        "has_punct": int(any(ch in string.punctuation for ch in word)),
        "is_short_word": int(len(lower) <= 3),
        "is_long_word": int(len(lower) >= 8),
    }

    # --- LETTER PRESENCE ---
    for vowel in "aeiou":
        feats[f"num_{vowel}"] = lower.count(vowel)

    # --- FILIPINO MORPHOLOGY ---
    for pre in FILIPINO_PREFIXES:
        feats[f"starts_{pre}"] = int(lower.startswith(pre))
    for suf in FILIPINO_SUFFIXES:
        feats[f"ends_{suf}"] = int(lower.endswith(suf))
    feats["has_ng"] = int("ng" in lower)
    feats["has_mga"] = int("mga" in lower)
    feats["has_reduplication"] = int(bool(re.search(r"(.+)-\1", lower)))  # araw-araw
    feats["has_reduplication_flex"] = int(bool(re.search(r"([a-z]{2,})\1", lower)))  # haha, sige-sige

    # --- ENGLISH MORPHOLOGY ---
    for suf in ENGLISH_SUFFIXES:
        feats[f"ends_{suf}"] = int(lower.endswith(suf))
    for cluster in ENGLISH_CLUSTERS:
        feats[f"has_{cluster}"] = int(cluster in lower)

    # --- SHORT-WORD DISAMBIGUATION ---
    feats["is_common_eng_short"] = int(lower in COMMON_EN_SHORTS)
    feats["is_common_fil_short"] = int(lower in COMMON_FIL_SHORTS)

    # --- SEMANTIC HINTS (mini dictionaries) ---
    feats["is_known_eng"] = int(lower in COMMON_ENG_WORDS)
    feats["is_known_fil"] = int(lower in COMMON_FIL_WORDS)

    # --- STRUCTURAL / SYMBOLIC ---
    feats["is_acronym"] = int(word.isupper() and len(word) <= 5 and word.isalpha())
    feats["is_camelcase"] = int(bool(re.match(r"[A-Z][a-z]+[A-Z]", word)))
    feats["has_mention"] = int("@" in word)
    feats["has_hashtag"] = int("#" in word)
    feats["has_url"] = int("http" in lower or ".com" in lower)
    feats["is_symbolic"] = int(any(ch in string.punctuation for ch in word) or any(ch.isdigit() for ch in word))

    # --- HYBRID RULES ---
    feats["ends_with_vowel_and_long"] = int(lower[-1:] in "aeiou" and len(lower) >= 5)

    # --- CHARACTER-LEVEL N-GRAMS ---
    for i in range(len(lower) - 1):
        feats[f"bi_{lower[i:i+2]}"] = 1
    for i in range(len(lower) - 2):
        feats[f"tri_{lower[i:i+3]}"] = 1

    # --- CONTEXT FEATURES ---
    
    if prev_word:
        feats["prev_ends_vowel"] = int(prev_word[-1].lower() in "aeiou")
        feats["prev_is_capitalized"] = int(prev_word[0].isupper()) if len(prev_word) > 0 else 0
    else:
        feats["prev_ends_vowel"] = 0
        feats["prev_is_capitalized"] = 0

    if prev_pred:
        feats[f"prev_pred_{prev_pred}"] = 1
    else:
        feats["prev_pred_NONE"] = 1

    return feats
