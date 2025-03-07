import nltk
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import api

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

words_to_remove = list(stopwords.words('english'))
to_remove_regex = r'\\n|\\|aita|“|”'

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

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def write_to_csv(rows):
    with open('data.csv', 'w') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows(rows)

default_preprocessor = TfidfVectorizer().build_preprocessor()
def custom_preprocessor(text):
    text = decontracted(text)
    text = text.translate(str.maketrans("","", string.punctuation))
    text = default_preprocessor(text)
    return text

labels = api.labels
samples = api.posts

rows = []

vectorizer = TfidfVectorizer(min_df=0.45, preprocessor=custom_preprocessor, stop_words="english", tokenizer=LemmaTokenizer())

X = vectorizer.fit_transform(samples).toarray()

for i in range(len(X)):
    row = [val for val in X[i]] + [labels[i]]
    rows.append(row)

write_to_csv(rows)
