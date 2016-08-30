__author__ = 'Harsh'
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time
import datetime
import nltk # If you have not downloaded the nltk all packages you can download it by nltk.download() command.

# Set start time of the date to find the freshness of each review
start_date = datetime.datetime(2013,3,13)

# Function to count no of lines each review has
def noOflines(text):
    return text.count('\n')

def noOfWords(text):
    return len(nltk.word_tokenize(text.decode('utf-8')))

def preprocessReviews(review_test_text):
	tokenizer = WordPunctTokenizer() # Prepare the tokenizer
	stemmer = PorterStemmer() # Prepare the stemmer
	stopset = set(stopwords.words('english')) # Prepare stopwords
	
	stemReviews = []
	for i in range(len(review_training_text)):
	    stemReviews.append(
	        [stemmer.stem(word.decode('latin_1')) for word in [w for w in
	            tokenizer.tokenize(review_training_text[i].lower())
	                if (w not in stopset)]
	        ]
	    ) #stem each review and remove the stopwords

	return stemReviews

def extractReviewFeatures(review_test):
	review_test = review_test.set_index('review_id')
	review_test = review_test.drop(['type'], axis=1)
	review_test['text'] = review_test['text'].fillna("")
	review_test['length of review'] = review_test['text'].apply(len)
	review_test['no of lines'] = review_test['text'].apply(noOflines)
	review_test['no of words'] = review_test.text.map(noOfWords)

	review_test['text'] = preprocessReviews(review_test['text'].tolist())

	review_test['stem review text length'] = review_test['text'].apply(len) # find the stemmed review's length
	review_test['text'] = [' '.join(text) for text in review_test['text']] # join all the text in reviews

	review_test['date'] = review_test.date.map(pd.to_datetime)  # parse the date to actual datetime
	review_test['date'] = start_date - review_test['date']  # find the freshness of review
	review_test['date'] = review_test['date'].apply(lambda p: p/np.timedelta64(1,'D'))  # convert the difference into days
	review_test = review_test.drop(['text'], axis=1)

	return review_test

def main():
	review_test = pd.read_csv('..\yelp_test_set\yelp_test_set_review.csv')
	review_test_extracted_features = extractReviewFeatures(review_test)
	review_test_extracted_features.to_csv('..\Processed Features\Review_test_features.csv')

if __name__ == '__main__':
	main()