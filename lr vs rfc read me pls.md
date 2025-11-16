# why we ended up using random forest instead of logistic regression

when we first tried using logistic regression (lr) for pinoybot, it worked fine for simple text, specifically ones that we're really straight forward. but once we started dealing with real code-switching and added more complex features, lr just couldn’t handle it anymore. 

but lr removes unimportant weights right?

- it does but it doesnt remove impactful ones :)))))))))))))))

lr is a linear model. that means it literally just adds up weights for each feature. so if a word looks english (like it has “ff”, “air”, “tion”, “cof”, etc.), lr pushes it toward eng. if the word has a filipino prefix (like “mag-”, “nag-”, “pa-”), lr pushes it toward fil. but it can’t combine these properly, especially for hybrid words.

example: **“nag-coffee”**

lr sees:
- starts_nag → filipino
- but english clusters like “cof”, “ff”, “ee”, etc. → english
- hyphen (punctuation) → others
- lots of english-like n-grams → english

if u think about it, lr will give impactful weights for these features and if u count them, the number of impactful features lean more towards english.

lr just adds everything and usually ends up predicting eng, even though we instantly know it's fil. the same thing happens for words like:

- “maghaircut”
- “pa-meet”
- “gustong-gusto”
- “sigee”
- “lol”
- punctuation-only tokens
- named entities like “starbucks”

lr tries, but it doesn’t understand *logic*, it only understands linear weights.

### why random forest works better

random forest (rfc) learns **rules**, not just weights. it's made of a bunch of small decision trees. each tree learns simple “if/else” patterns like:

if starts_mag → fil
if starts_nag and has_hyphen → fil
if reduplication → fil
if pure punctuation → oth
if english-looking root but has filipino prefix → still fil

*take note* it rfc actually gets a part of the features (bagging) and labels and makes a decision tree of its own, it does it n times (depending on what we set for n_estimators) and will end up with a certain label


then all the trees vote, and whichever label gets the most votes wins. this kind of structure works way better for messy filipino code-switching because the language is full of exceptions, hybrids, slang, and weird spellings. lr can’t capture that, but rfc does it naturally, since it checks almost everything (bootstrapping).



### what rfc handles that lr struggles with

rfc is good at:
- non-linear patterns
- feature combinations (prefix + root + hyphen)
- hybrid words
- filipino affixes attached to english roots
- reduplication (like “gustong-gusto” or informal “sigee”)
- elongated slang
- named entities
- punctuation-only tokens
- words that look english but are used in a filipino way

lr only does linear sums. rfc actually builds decision paths.


### tl;dr

lr just adds weights. rfc learns rules.  
code-switching needs rules, not linear math.  
so yeah, random forest wins for this kinds of word.
try to use ur own sentences para we know if we want to switch
