Why we ended up using random forest instead of logistic regression
When we first tried using logistic regression (LR) for Pinoybot, it worked fine for simple text, specifically ones that were really straightforward. But once we started dealing with real code-switching and added more complex features, LR just couldn’t handle it anymore.

But LR removes unimportant weights, right?

It does, but it doesn't remove impactful ones.
LR is a linear model, which means it literally just adds up weights for each feature. So if a word looks English (like it has “ff”, “air”, “tion”, “cof”, etc.), lr pushes it toward English. If the word has a Filipino prefix (like “mag-”, “nag-”, “pa-”), lr pushes it toward fil. But it can’t combine these properly, especially for hybrid words.

example: “nag-coffee”

LR sees:

starts_nag → filipino
but English clusters like “cof”, “ff”, “ee”, etc. → English
hyphen (punctuation) → others
lots of English-like n-grams → English
if u think about it, LR will give impactful weights for these features, and if u count them, the number of impactful features leans more towards English.

LR just adds everything and usually ends up predicting eng, even though we instantly know it's fil. The same thing happens for words like:

“maghaircut”
“pa-meet”
“gustong-gusto”
“sigee”
“lol”
punctuation-only tokens
named entities like “starbucks”
LR tries, but it doesn’t understand logic, it only understands linear weights.

Why random forest works better
random forest (RFC) learns rules, not just weights. It's made of a bunch of small decision trees. each tree learns simple “if/else” patterns like:

if starts_mag → fil if starts_nag and has_hyphen → fil if reduplication → fil if pure punctuation → oth if English-looking root but has a Filipino prefix → still Filipino

take note it RFC actually gets a part of the features (bagging) and labels and makes a decision tree of its own, it does it n times (depending on what we set for n_estimators) and will end up with a certain label

Then all the trees vote, and whichever label gets the most votes wins. This kind of structure works way better for messy Filipino code-switching because the language is full of exceptions, hybrids, slang, and weird spellings. LR can’t capture that, but RFC does it naturally, since it checks almost everything (bootstrapping).

What RFC handles that LR struggles with
RFC is good at:

non-linear patterns
feature combinations (prefix + root + hyphen)
hybrid words
Filipino affixes attached to English roots
reduplication (like “gustong-gusto” or informal “sigee”)
elongated slang
named entities
punctuation-only tokens
words that look English but are used in a Filipino way
LR only does linear sums. RFC actually builds decision paths.

tl;dr
lr just adds weights. RFC learns rules.
Code-switching needs rules, not linear math.
