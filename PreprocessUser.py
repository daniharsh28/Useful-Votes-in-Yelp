__author__ = 'Harsh'
import pandas as pd
import numpy as np

####################################################
#Load user data into the python DataFrame
###################################################
user_training_frame = pd.read_csv('E:\Fall 2014\Social Media Mining\yelp_training_set\yelp_training_set_user.csv', header = 0, index_col = 'user_id')
user_test_frame = pd.read_csv('E:\Fall 2014\Social Media Mining\yelp_training_set\yelp_training_set_user.csv', header = 0, index_col = 'user_id')
###################################################################
user_test_frame['pp'] = pd.Series(np.ones(user_test_frame.shape[0]), index= user_test_frame.index)
user_training_frame['pp'] = pd.Series(np.zeros(user_training_frame.shape[0]), index= user_training_frame.index)
################################################################
# Combine training and test data
user_final_frame = user_training_frame.combine_first(user_test_frame)
######################################################
# Features will be ration of no.of useful_votes over no.of reviews
user_final_frame['votes_useful_over_no_of_reviews'] = user_final_frame.votes_useful/(user_final_frame.review_count+1)#Added 1 to avoid divison by zero
user_final_frame = user_final_frame.drop(['name','type'], axis=1)

user_final_frame.to_csv('E:\Fall 2014\Social Media Mining\Processed Features\user_features.csv')