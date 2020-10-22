import numpy as np
import pandas as pd
import chardet
import matplotlib.pyplot as plt
#import used for preprocessing.
import spacy
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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
tok_rev = []
idx = 0
# Process is currently too demanding, but will work and run on a subset of the data. 
for review in nlp.pipe(reviews[0:1000]):
    # Passes each token into function to exclude all punctuation, spaces, and stop words.
    tokens = [token.text for token in review if token_filter(token) ]
    # Appends list of approved tokens to the tokenized review list.
    tok_rev.append(tokens)

# Create dictionary to pass into new dataframe.
rev_dic = {'Rating' : freshness[0:1000], 'text_data' : tok_rev}
rev_df = pd.DataFrame(rev_dic)

# Separate container for the entire vocabulary of the data set. 
vocab = []

for token_list in tok_rev:
    for word in token_list:
        vocab.append(word)

unique_words = np.unique(vocab, return_counts = True)[0]
unique_counts = np.unique(vocab, return_counts = True)[1]

# Apply numerical labels to Ratings for Naive Bayes classification.

rev_df['Rating'] = rev_df['Rating'].replace('fresh', 0)
rev_df['Rating'] = rev_df['Rating'].replace('rotten', 1)

# plt.bar(unique_words, unique_counts, tick_label = unique_words)
# plt.show()

# Declare index so we can append columns beginning at position 2 in the dataframe. (After Rating and Text_data)
idx = 2
# Add a column for each unique word for frequency matrix.
for word in unique_words:
    rev_df.insert(idx, word, np.zeros(1000), allow_duplicates = True)
    idx += 1

for index in rev_df.index:
    for word in rev_df.loc[index, "text_data"]:
        rev_df.loc[index, word] += 1.0



review_data['Freshness'] = np.where(review_data['Freshness']=='Freshness',1,0)


X_train, X_test, y_train, y_test = train_test_split(review_data['Review'][0:10000], 
                                                    review_data['Freshness'][0:10000], 
                                                    test_size = 0.2, 
                                                    random_state = 42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
predicted = classifier.predict(X_test_tfidf)
acc = metrics.accuracy_score(y_test, predicted)

print('accuracy: ', acc*100)