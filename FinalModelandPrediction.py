__author__ = 'Harsh'
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import operator
from pylab import *

def load_data():

	review = pd.read_csv('..\\Processed Features\\review_training_features.csv')
	user = pd.read_csv('..\\Processed Features\\user_features.csv').set_index('user_id') # Load the user data from
	business = pd.read_csv('..\\Processed Features\\Business_features.csv').set_index('business_id') # Load the buisness data
	checkin = pd.read_csv('..\\Processed Features\\Checkin_features.csv').set_index('business_id') # Load the chcekin data
	business = business.join(checkin) # Join the checkin and buisness data on business id
	
	review = review.join(user, on='user_id', rsuffix= '_user') # Join review and user data
	review = review.join(business, on='business_id', rsuffix= '_business') 

	review = review.drop(['business_id','user_id', 'city','open'], axis = 1)
	review = review.set_index('review_id')
	review = review.fillna(0)

	return review

def extractDerivedFeatures(review):

	review['user_review_delta'] = abs(review['average_stars'] - review['stars'])
	review['user_review_delta'] = review['user_review_delta'].fillna(0)
	review['business_review_delta'] = abs(review['stars_business'] - review['stars'])
	review['business_review_delta'] = review['business_review_delta'].fillna(0)
	return review

def getTrainValidationSet(review):

	review_train = review._slice(slice(0,1000),0)
	review_valid = review_train.ix[:int(len(review_train)*0.1),:]
	review_train = review_train.ix[set(review_train.index) - set(review_valid.index),:]
	return review_train, review_valid

#Root mean square log error
def rmsle(train,test): # Define the Root Mean Square Logarithm Error
    return np.sqrt(np.mean((pow(np.log(test+1) - np.log(train+1),2))))

def getGradBoost():

	# Set up the gradient boosting regressor
	gradientboost = GradientBoostingRegressor(n_estimators=400, max_depth= 7, random_state=7)
	return gradientboost

def train(review, labels, clf):
	clf.fit(review,labels)

def main():

	review = load_review_data()
	review = extractDerivedFeatures(review) # Use Derived features and add it to review frame.
	review_train, review_valid = getTrainValidationSet(review)

	# Drop the class label
	labels = review_train.drop(['votes_useful'], axis=1)
	clf = getGradBoost()
	train(review, labels, clf)
	
	cap_result = gradientboost.predict(review_valid.drop(['votes_useful'], axis=1))
	print 'The rmsle is ' + str(rmsle(cap_result, review_valid['votes_useful']))

if __name__ == '__main__':
	main()