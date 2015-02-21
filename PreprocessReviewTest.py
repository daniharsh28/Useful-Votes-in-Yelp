__author__ = 'Harsh'
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time
import datetime
import nltk # If you have not downloaded the nltk all packages you can download it by nltk.download() command.

####################################
# Set start time of the date to find the freshness of each review
start_date = datetime.datetime(2013,3,13)
##########################################
# Function to count no of lines each review has
def noOflines(text):
    return text.count('\n')

def noOfWords(text):
    return len(nltk.word_tokenize(text.decode('utf-8')))
#####################################
# Now load the review data into panda DataFrame
review_training = pd.read_csv('E:\Fall 2014\Social Media Mining\yelp_test_set\yelp_test_set_review.csv')
review_training = review_training.set_index('review_id')
review_training = review_training.drop(['type'], axis=1)
review_training['text'] = review_training['text'].fillna("")
review_training['length of review'] = review_training['text'].apply(len)
review_training['no of lines'] = review_training['text'].apply(noOflines)
print 'Finding no of words'
review_training['no of words'] = review_training.text.map(noOfWords)
tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()
########################################################
# remove the stopwords and find the stemming root of each word
stopset = set(stopwords.words('english'))
stemReviews = []
print 'Stemming and stopword removal started'
for i in range(len(review_training)):
    stemReviews.append(
        [stemmer.stem(word.decode('latin_1')) for word in [w for w in
            tokenizer.tokenize(review_training.ix[i,'text'].lower())
                if (w not in stopset)]
        ]
    )
print 'Stemming and stopwords removal completed!'
review_training['text'] = stemReviews # append stemmed reviews
review_training['stem review text length'] = review_training['text'].apply(len) # find the stemmed review's length
review_training['text'] = [' '.join(text) for text in review_training['text']] # join all the text in reviews
#review_training['text'] = review_training['text'].apply(lambda x: x.decode('utf-8'))
review_training['date'] = review_training.date.map(pd.to_datetime)  # parse the date to actual datetime
review_training['date'] = start_date - review_training['date']  # find the freshness of review
review_training['date'] = review_training['date'].apply(lambda p: p/np.timedelta64(1,'D'))  # convert the difference into days
review_training = review_training.drop(['text'], axis=1)
print 'Writing the data into csv'
review_training.to_csv('E:\Fall 2014\Social Media Mining\Processed Features\Review_test_features.csv')


