🧠 How Our Model Works (and Why We Ended Up Using Logistic Regression)

So for this project, we tried a few different models — Decision Tree, Random Forest, Naive Bayes, and Logistic Regression — to see which one could best identify whether each word in a sentence is Filipino, English, or Other.

After a lot of testing, we decided to go with Logistic Regression, because it gave the most balanced and consistent results across validation and test data.

⚙️ How Logistic Regression Actually Works

Logistic Regression isn’t actually “regression” like predicting numbers — it’s a classifier.
In our case, it’s predicting the probability that a word is FIL, ENG, or OTH.

It works using the features we made in feature_utils.py — things like:

whether the word starts with mag or nag

if it ends with ing or ed

if it contains ng, tion, or mga

if it’s all caps, a number, or has punctuation

Each of those acts as a clue that hints at which language the word belongs to.

During training, the model assigns a weight to each clue.
For example:

starts_mag might strongly point to Filipino

ends_ing might point to English

has_ng could also suggest Filipino

So when you input a word like "magtraining", the model multiplies each feature by its weight, adds everything up, and converts it into probabilities — for example:

FIL: 0.85, ENG: 0.10, OTH: 0.05

Then it picks the label with the highest probability.
That’s literally how our final predictions are made.

🔁 What Happens During Training

When training starts, the model doesn’t know anything — all weights start randomly.
It predicts the wrong labels at first, checks how wrong it was, and slightly adjusts the weights.
This repeats over and over (we set max_iter=1000) until the weights stabilize and the predictions become accurate.

By the end, it has learned how strongly each feature affects each label —
for example, how much “ends_ing” contributes to being English versus Filipino.

💡 Why Logistic Regression Performed Better Than Decision Trees and Random Forests

We also tried Decision Trees and Random Forests, but Logistic Regression performed better for a few reasons.

Model	How It Works	Why It Didn’t Fit Our Data
Decision Tree	Makes strict “if-else” splits (e.g., if ends_ing=1 → English)	Overfits easily — memorizes the training data instead of learning general patterns
Random Forest	Combines many trees to reduce overfitting	Still struggles when features overlap or depend on each other
Logistic Regression	Uses all features together and finds weighted probabilities	Handles overlapping or dependent features smoothly

Filipino-English code-switching isn’t rule-based. Some Filipino words look English, and some English words borrow Filipino spelling. Because Decision Trees make hard boundaries, they can’t deal with that overlap. Logistic Regression, on the other hand, draws a soft decision boundary — it balances probabilities instead of committing to a single hard rule.

🤖 Why We Didn’t Use Naive Bayes

Naive Bayes assumes all features are independent, meaning it thinks each feature affects the label separately.
That’s fine for something like spam detection (“has free”, “has win”, “has money”), but it doesn’t make sense for language classification.

In our dataset, features are clearly dependent — they interact with each other.
For example:

starts_mag=1 (Filipino clue) and ends_ing=1 (English clue) appearing together might actually mean code-switched Filipino-English.

has_ng and has_mga often occur in the same word — they reinforce each other, not act independently.

Naive Bayes doesn’t account for that — it just multiplies independent probabilities.
So it oversimplifies our linguistic features and ends up guessing wrong more often.

Logistic Regression fixes that by combining all features together, giving each one a weight that’s learned in context with all the others.

⚖️ So, What Does “Dependent Features” Mean in Our Case?

It basically means some features only make sense when seen together.

Example:

“ends_ing” alone might mean English —
but if it’s also “starts_mag”, it’s probably a Filipino-English mixed word.

“is_capitalized” isn’t useful by itself —
but combined with “has_punct” or “has_digit”, it might hint at something like “OTH” (symbols, names, or expressions).

Because our linguistic clues rely on one another, they’re dependent.
That’s why a model that can handle relationships between features — like Logistic Regression — performs better than one that assumes independence (like Naive Bayes).

🏁 In Simple Terms

Logistic Regression looks at all clues together and learns how strong each one is.

Decision Trees and Random Forests rely on hard yes/no rules, which overfit easily.

Naive Bayes assumes clues don’t interact, which doesn’t match real language patterns.

Our features (prefixes, suffixes, capitalization, etc.) are dependent, so Logistic Regression makes more sense.