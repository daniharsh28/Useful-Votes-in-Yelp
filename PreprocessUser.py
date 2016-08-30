__author__ = 'Harsh'
import pandas as pd
import numpy as np

#Load user data into the python DataFrame

user_training_frame = pd.read_csv('../yelp_training_set/yelp_training_set_user.csv', header = 0, index_col = 'user_id')
user_test_frame = pd.read_csv('../yelp_test_set/yelp_test_set_user.csv', header = 0, index_col = 'user_id')

user_test_frame['pp'] = pd.Series(np.ones(user_test_frame.shape[0]), index= user_test_frame.index)
user_training_frame['pp'] = pd.Series(np.zeros(user_training_frame.shape[0]), index= user_training_frame.index)
user_final_frame = user_training_frame.combine_first(user_test_frame)

# We used ratio of number of userful votes to number of reviews as feature
user_final_frame['votes_useful_over_no_of_reviews'] = user_final_frame.votes_useful/(user_final_frame.review_count+1)#Added 1 to avoid divison by zero
user_final_frame = user_final_frame.drop(['name','type'], axis=1)

user_final_frame.to_csv('../processed_features/user_features.csv')