import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import api

words_to_remove = list(stopwords.words('english'))
to_remove_regex = r'\\n|\\|aita|“|”'
lemmatizer = WordNetLemmatizer()

# to limit amount of features we can adjust parameters min_df, max_features
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# modify min_df as needed
vectorizer = TfidfVectorizer(min_df=0.45)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won[’|']t", "will not", phrase)
    phrase = re.sub(r"can[’|']t", "can not", phrase)

    # general
    phrase = re.sub(r"n[’|']t", " not", phrase)
    phrase = re.sub(r"[’|']re", " are", phrase)
    phrase = re.sub(r"[’|']s", " is", phrase)
    phrase = re.sub(r"[’|']d", " would", phrase)
    phrase = re.sub(r"[’|']ll", " will", phrase)
    phrase = re.sub(r"[’|']t", " not", phrase)
    phrase = re.sub(r"[’|']ve", " have", phrase)
    phrase = re.sub(r"[’|']m", " am", phrase)
    return phrase

def preprocess(sample):
    text = sample.lower()

    text = re.sub(to_remove_regex, '', text)

    # removing punctuation
    text = text.translate(str.maketrans("","", string.punctuation))

    # expanding contractions
    text = decontracted(text)

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in words_to_remove])

    return text

def vectorize(samples):
    rows = []

    rows = [preprocess(sample) for sample in samples]

    X = vectorizer.fit_transform(rows)

    print(X.shape)

    return X.toarray()

def write_to_csv(rows):
    with open('data.csv', 'w') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows(rows)

labels = api.labels
samples = api.posts

rows = []

X = vectorize(samples)

for i in range(len(X)):
    row = [val for val in X[i]] + [labels[i]]
    rows.append(row)

write_to_csv(rows)
