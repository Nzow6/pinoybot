## How Logistic Regression Actually Works

Logistic Regression isn’t actually “regression” like predicting numbers (what you guys will usually here is like prediciting house prices or stock prices, AKA something continuous) — it’s a **classifier**.
In our case, it’s predicting the **probability** that a word is `FIL`, `ENG`, or `OTH`.

It works using the **features** we made in `feature_utils.py` which are things like:

* whether the word starts with `mag` or `nag`
* if it ends with `ing` or `ed`
* if it contains `ng`, `tion`, or `mga`
* if it’s all caps, a number, or has punctuation

Each of those acts as a **clue** that hints at which language the word belongs to.

During training, the model assigns a **weight** to each clue.
For example:

* `starts_mag` might strongly label to Filipino
* `ends_ing` might be labeled in English
* `has_ng` could also suggest Filipino in some cases

So when you input a word like `"magtraining"`, the model multiplies each feature by its weight, adds everything up, and converts it into probabilities (there's actually a math formula for this, but that's not our job) — for example:

> FIL: 0.85, ENG: 0.10, OTH: 0.05

Then it picks the label with the highest probability.
That’s literally how our final predictions are made.
**This is technically done by the Softmax Function**

---

## What Happens During Training

When training starts, the model doesn’t know anything — all the weights start off randomly. It predicts wrong at first, checks how far off it was, and slightly adjusts those weights. This repeats over and over (we set `max_iter=1000`, which we can modify — higher iterations just give it more chances to find the best weights for each feature) until everything stabilizes and the predictions start making sense.

By the end, the model has learned how strongly each feature affects each label — for example, how much “ends_ing” contributes to a word being English versus Filipino.

CODE PART FOR REFERENCE IN NEXT PARAGRAPH: `log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`

But here’s the main reason why it jsut fits better with our mco and features: it doesn’t just learn blindly. Our model uses an **L2 penalty** (default in scikit-learn, that's why I didn't specify it), which is like a balancing tool that keeps the model from over-relying on any single feature. It doesn’t only shrink useless features — it also tones down the ones that get *too strong*. For instance, if “starts_mag” gets a massive weight because it often shows up in Filipino words, L2 will smooth it out a bit so the model doesn’t make everything with “mag” automatically Filipino. Instead, it spreads attention across other features like “ends_ing” or “contains_th” that also matter.

This keeps the model well-balanced and stable — it focuses on patterns that consistently matter, while ignoring random noise or overly dominant features. In short, L2 makes sure the model learns **smart confidence**, not **blind certainty**.

---

## Why Logistic Regression Performed Better Than Decision Trees and Random Forests (this part table is prompt engineered haha)

We also tried Decision Trees and Random Forests, but Logistic Regression performed better for a few reasons.

| Model                   | How It Works                                                    | Why It Didn’t Fit Our Data                                                         |
| ----------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Decision Tree**       | Makes strict “if-else” splits (e.g., if `ends_ing=1` → English) | Overfits easily — memorizes the training data instead of learning general patterns |
| **Random Forest**       | Combines many trees to reduce overfitting                       | Still struggles when features overlap or depend on each other                      |
| **Logistic Regression** | Uses all features together and finds weighted probabilities     | Handles overlapping or dependent features smoothly                                 |

Filipino-English code-switching isn’t rule-based. Some Filipino words look English, and some English words borrow Filipino spelling. Because Decision Trees make hard boundaries, they tend to create strict “if-then” rules. For example, if the tree learns that words ending with “er” are English, it might automatically classify anything ending with “er” (like “taga-gather”) as English, even when it’s being used as part of a Filipino sentence or being code-switched. Once the tree makes that decision, it doesn’t consider other clues (like the presence of “taga” or “na”) — it just follows that single rule all the way down.

Logistic Regression, on the other hand, doesn’t commit to one hard rule. It considers all features together and assigns a probability to each class. So in the same example, it might recognize that “ends with er” suggests English, but “starts with taga” suggests Filipino — and then combine those signals instead of picking one over the other. This “soft” decision-making allows it to handle code-switched words better, where boundaries between Filipino and English aren’t always clear.

---

## Why We Didn’t Use Naive Bayes

Naive Bayes assumes all features are **independent**, meaning it thinks each feature affects the label separately.
That’s fine for something like spam detection (“has free”, “has win”, “has money”), but it doesn’t make sense for language classification.

In our dataset, features are clearly **dependent** — they interact with each other.
For example:

* `starts_mag=1` (Filipino clue) and `ends_ing=1` (English clue) appearing together might actually mean **code-switched Filipino-English**.
* `has_ng` and `has_mga` often occur in the same word — they reinforce each other, not act independently.

Naive Bayes doesn’t account for that — it just multiplies independent probabilities.
So it oversimplifies our linguistic features and ends up guessing wrong more often.

Logistic Regression fixes that by **combining** all features together, giving each one a weight that’s learned in context with all the others.

---

## So, What Does “Dependent Features” Mean in Our Case?

It basically means some features only make sense **when seen together**.

Example:

* “ends_ing” alone might mean English —
  but if it’s also “starts_mag”, it’s probably a Filipino-English mixed word.
* “is_capitalized” isn’t useful by itself —
  but combined with “has_punct” or “has_digit”, it might hint at something like “OTH” (symbols, names, or expressions).

Because our extracted features rely on one another, they’re **dependent**.
That’s why a model that can handle relationships between features — like Logistic Regression — performs better than one that assumes independence (like Naive Bayes).

---

## TLDR:

* Logistic Regression looks at **all clues together** and learns how strong each one is.
* Decision Trees and Random Forests rely on **hard yes/no rules** (if statements), which overfit easily.
* Naive Bayes assumes **clues don’t interact**, which doesn’t match real language patterns.
* Our features (prefixes, suffixes, capitalization, etc.) are **dependent**, so Logistic Regression makes more sense.







