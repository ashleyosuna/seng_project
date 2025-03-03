import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

words_to_remove = list(stopwords.words('english'))
to_remove_regex = r'\\n|\\|aita|“|”'
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

example = 'AITA for wanting to live alone? I(27f) told my mom(62f) that I think I want to live alone now she said I’m being a bad daughter and that I should let her live out the rest of her life with me. \n\nMy mom was a single mom and did An amazing job I have zero childhood trauma. I never struggled in life thanks to her. I didn’t like my school?shed move to a different zip code so I could go where I wanted. I was really depressed? I never got in trouble for being a bad teen. I have zero negative comments about her she’s amazing. \nWhen I became an adult she started to struggle financially so I started helping out with bills. We’re at a point where I pay 60% she pays 40% of rent and I pay everything else. She can’t really afford to live on her own we’re kinda like bestfriend roommates. but I feel I should have the opportunity to finally be an independent adult and live on my own. My mom has heart issues so her response is “I won’t be here much longer”. \nSo AMTA for wanting to live alone?'
example2 = """
AITA for cooking food for my female friend? 
So (27m) I've been dating my gf (26f) for a year. Things have been going well. I have a friend Rhea(26f), been friends since college.



 Rhea is a vegetarian. She has never tried it growing up and doesn't want to. and she's not "a bitch about it",as my girl says. She has no problem with someone eating it beside her or smn. She just doesn't eat it.


Onto the topic, I love cooking and I occasionally invite our friends over for dinne. Whenever I host dinner, Rhea always checks in with me if I'm gonna be cooking something vegetarian or should she just bring something. I always do as I like it too. I'll admit, sometimes when it's not on the menu, I still do because I don't want someone to feel left out and as a host it's my responsibility. 


This thanksgiving, a friend hosted dinner and Rhea brought a dish that everyone liked and the host said she didn't have to as she cooked for her as well. My girlfriend said that she never brought a dish over at my place. Rhea told her that I already do my best to accommodate. Well this pissed my girlfriend off and she left the table. I went after her, she started screaming about how I never told her (not true, I have mentioned this,multiple times) Basically made everyone uncomfortable and Rhea left after saying she was sorry if her eating habits were causing trouble. 


 When we went back to the table, the mood was sour and when we were leaving my friends got me alone and told me how it's my girl's fault for ruining the dinner. I agree, whatever the issue was, could've been discussed in private. I did try to talk about it with her again, but she said she's good and doesn't wanna talk about it anymore.


I'm hosting a dinner again as a friend of ours is coming back from abroad. My gf said there would be no vegetarian food this time as "she doesn't wanna see me prioritizing some other girl over her again." 
I again tried to talk about this and she says she can't believe I'm cooking a whole new dish for some girl and how it's disrespectful to her. I told her she's a friend and everyone wants her here. This turned into screaming again that ended with her saying she knows I'm cheating on her with Rhea and calling her names. I've tried to comfort her on this, have tried to talk if something else has happened that makes her feel this way but I got nothing more than "you're not cooking for her, end of topic".


I don't understand. I've done nothing to show that I have any interest in my friend. There's no history between us if anyone's wondering. I host these dinners usually once a month and that's the only time she's over. We don't really hang out with just the two of us and she's never done anything that'll make my girlfriend uncomfortable or I'd have shut it down myself. 


That thanksgiving dinner was the first time this was brought up. I need an actual answer from her other than "I don't want you to".So AITA?
"""

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

    print(text)
    return text

def vectorize(samples):
    rows = []

    rows = [preprocess(sample) for sample in samples]

    X = vectorizer.fit_transform(rows)

    return X, labels

# TESTING
# vectorize([example, example2])
