import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text)->str:
    CLEANR = re.compile('<.*?>') 
    stopwords_rus = stopwords.words('russian')

    text = re.sub(CLEANR, '', text)
    text = "".join([char for char in text if char not in string.punctuation+'»«—'])
    res = [w.lower() for w in nltk.word_tokenize(text, language='russian')]
    res = [w for w in res if w not in stopwords_rus]
    res = [w for w in res if not w.isnumeric()]
    res = [w.replace('\n','') for w in res]
    return ' '.join(res)

def preprocess_arr(arr):
    new_data = []
    for a in arr:
        new_data.append(preprocess_text(a))
    return new_data
