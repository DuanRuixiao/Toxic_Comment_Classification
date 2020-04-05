import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import os

train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# extract text in train and test data and Concatenate together
X_train_text = train['comment_text']
X_test_text = test['comment_text']
all_text = pd.concat([X_train_text, X_test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True, strip_accents='unicode',
    analyzer='word', token_pattern=r'\w{1,}',
    stop_words='english', ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(all_text)
X_train = word_vectorizer.transform(X_train_text)
X_test = word_vectorizer.transform(X_test_text)

# Apply logistic regression
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for list_class in list_classes:
    y_train = train[list_class]
    classifier = LogisticRegression(C=0.1, solver='saga')
    
    cv_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(list_class, cv_score))

    classifier.fit(X_train, y_train)
    submission[list_class] = classifier.predict(X_test)
    
print('Total CV score is {}'.format(np.mean(scores)))

os.chdir("../outputs")
submission.to_csv('submission_lr.csv', index=False)
