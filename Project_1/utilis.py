
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk import sent_tokenize
from nltk import word_tokenize

import tensorflow as tf
import tensorflow_hub as hub

stop_words = stopwords.words()

def cleaning(text):        
    
    text = re.sub(r"won't", "will not",text)
    text = re.sub(r"can\'t", "can not",text)
    text = re.sub(r"n\'t", " not",text)
    text = re.sub(r"\'re", " are",text)
    text = re.sub(r"\'s", " is",text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'t", " not",text)
    text = re.sub(r"\'ve", " have",text)
    text = re.sub(r"\'m", " am",text)
    text = re.sub('RT'," ", text)
    

    text = re.sub("\$RESERVED\$ \$MENTION\$|\$NUMBER\$|\$MENTION\$",' ', text)
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('[’“”…]', ' ', text)   
    
    
    text = re.sub('user'," ", text)
    text = re.sub('url'," ", text)
    
    text = re.sub('reserved'," ", text)
    
    
    
    
    
    
    text = re.sub("[^A-Za-z]",' ',text)
    

    # removing the emojies               
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)   
    
    
    
    
    
    
    return text




    
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(text):
    return model(text)