import nltk 
import numpy as np
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Translates sentences into vectors | Changes strings into arrays with positional value
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Simplifies the words by turning them into their root | ex. 'organize', 'organizing', 'organizes' -> 'organ'
def stem(word):
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, all_words):
    """ Example showing how bag_of_words works
    sentence = ["hello", "how", "are", "you"]
    words = ["hi","hello","I","you","bye","thank","cool"]
    bog =   [  0,    1,    0,   1,    0,      0,     0  ]
    
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype = np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
            
    return bag

