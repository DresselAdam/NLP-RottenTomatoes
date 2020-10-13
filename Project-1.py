import numpy as np
import pandas as pd
import chardet
import matplotlib as plt
#import used for Lemmatization
import spacy

#Function used for preprocessing text data from csv
def token_filter(token):
    return not (token.is_punct | token.is_space | token.is_stop | len(token.text) <= 4)

# Loading lemma object for English.
nlp = spacy.load('en_core_web_sm')

'''
Code used when initially determining encoding type. Commented out due to lengthy runtime.
with open('rt_reviews.csv', 'rb') as line:
    enc_type = chardet.detect(line.read())

print(enc_type)
'''

# Reading csv with pandas
review_data = pd.read_csv('rt_reviews.csv', encoding = 'ISO-8859-1', index_col=False)

# Assigning each attribute to a new variable
freshness = review_data['Freshness']
reviews = review_data['Review'].tolist()


# Holds tokenized text of the reviews from the csv data.
filtered_vocab = []
idx = 0
# Process is currently too demanding, but will work and run on a subset of the data. 
for review in nlp.pipe(reviews[0:1000]):
    # Passes each token into function to exclude all punctuation, spaces, and stop words.
    tokens = [token.lemma_ for token in review if token_filter(token)]
    # Appends list of approved tokens to the vocabulary.
    filtered_vocab.append(tokens)
    print(freshness[idx],filtered_vocab[idx])
    idx += 1