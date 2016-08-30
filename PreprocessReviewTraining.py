__author__ = 'Harsh'
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time
import datetime
import nltk # If you have not downloaded the nltk all packages you can download it by nltk.download() command.

# Set start time of the date to find the freshness of each review, emprirically determined by looking at the data
start_date = datetime.datetime(2013,1,20)

# Function to count no of lines each review has
def noOflines(text):
    return text.count('\n')

def noOfWords(text):
    return len(nltk.word_tokenize(text.decode('utf-8')))

def preprocessReviews(review_training_text):
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

def extractReviewFeatures(review_training):
	review_training = review_training.set_index('review_id')
	review_training = review_training.drop(['type','votes_funny','votes_cool'], axis=1) # Drop type, funny votes and cool votes from review data 
	review_training['text'] = review_training['text'].fillna("") # impute the review data
	review_training['length of review'] = review_training['text'].apply(len) # Find review length
	review_training['no of lines'] = review_training['text'].apply(noOflines) # Find no of lines from review length
	review_training['no of words'] = review_training.text.map(noOfWords) # Find no of words in review

	review_training['text'] = preprocessReviews(review_training['text'].tolist()) # Preprocess review training text data
	
	review_training['stem review text length'] = review_training['text'].apply(len) # Stem Review Length
	review_training['date'] = review_training.date.map(pd.to_datetime)
	review_training['date'] = start_date - review_training['date'] # Find the freshess of the review
	review_training['date'] = review_training['date'].apply(lambda p: p/np.timedelta64(1,'D')) # Compute the days from timedelta
	review_training = review_training.drop(['text'], axis=1)
	
	return review_training

def main():
	# Load the training data
	review_training = pd.read_csv('..\yelp_training_set\yelp_training_set_review.csv')
	review_traning_extracted_features = extractReviewFeatures(review_training)
	review_traning_extracted_features.to_csv('..\Processed Features\Review_training_features.csv')

if __name__ == '__main__':
	main()