import numpy as np
import pandas as pd
import chardet
import matplotlib as plt
#import used for Lemmatization
import spacy

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
review = review_data['Review']

# Code for lemmatization
lemm_review = nlp(review[0])
for token in lemm_review:
    print(token.text, token.pos_)