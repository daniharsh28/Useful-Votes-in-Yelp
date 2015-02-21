__author__ = 'Harsh'
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import operator
from pylab import *
#########################################################################
# Load the user data from
user = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\user_features.csv').set_index('user_id')
# Load the buisness data
business = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\Business_features.csv').set_index('business_id')
# Load the chcekin data
checkin = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\Checkin_features.csv').set_index('business_id')
# Join the checkin and buisness data on business id
business = business.join(checkin)
#####################################
#Load the review data
review = pd.read_csv('E:\Fall 2014\Social Media Mining\Processed Features\\review_training_features.csv')
#Join user and review data
review = review.join(user, on='user_id', rsuffix= '_user')
# Join user and business data
review = review.join(business, on='business_id', rsuffix= '_business')
review = review.drop(['business_id','user_id', 'city','open'],axis = 1)
review = review.set_index('review_id')
review = review.fillna(0)
review['delta of user review'] = abs(review['average_stars'] - review['stars'])
review['delta of user review'] = review['delta of user review'].fillna(0)
review['delta of business review'] = abs(review['stars_business'] - review['stars'])
review['delta of business review'] = review['delta of business review'].fillna(0)
print review
#review.to_csv('E:\Fall 2014\Social Media Mining\Processed Features\Final_Features.csv')
##############################################
# Differentiate training and validation data
review_train = review._slice(slice(0,1000),0)
review_valid = review_train.ix[:int(len(review_train)*0.1),:]
review_train = review_train.ix[set(review_train.index) - set(review_valid.index),:]
print 'Size of training set %i' % len(review_train)
print 'Size of validation set %i' % len(review_valid)
review_without_useful = review_train.drop(['votes_useful'], axis=1)
######################################
#Root mean square log error
def rsmle(train,test): # Define the Root Mean Square Logarithm Error
    return np.sqrt(np.mean((pow(np.log(test+1) - np.log(train+1),2))))
################################################

# Set up the gradient boosting regressor
gradientboost = GradientBoostingRegressor(n_estimators=400, max_depth= 7, random_state=7)
#Remove no of useful votes

print 'Time to train the regressor'
gradientboost.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete time for predictions!'
print review_without_useful.columns.values
feature_importance = gradientboost.feature_importances_
print feature_importance
Feature_importance_dict = dict(zip(review_without_useful.columns.values,feature_importance))
cap_result = gradientboost.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle is ' + str(rsmle(cap_result, review_valid['votes_useful']))
Feature_importance_dict = (sorted(Feature_importance_dict.items(), key=operator.itemgetter(1),reverse=True)[:10])
Feature_importance_dict = dict(Feature_importance_dict)
figure(1)
barh(arange(len(Feature_importance_dict)),Feature_importance_dict.values(), align='center')
yticks(arange(len(Feature_importance_dict)),Feature_importance_dict.keys())
xlabel('Feature importances')
show()
#cap_result_approximated = cap_result.astype(int)
#l = pd.Series(cap_result_approximated)
#l = l.value_counts().to_dict()
#print l
#plt.bar(range(len(l)), l.values() )
#plt.xticks(range(len(l)))
#plt.xlabel('No. of useful votes')
#plt.ylabel('Review count')
#plt.show()

review_valid_plot = review_valid['votes_useful']
l1 = review_valid_plot.value_counts().to_dict()
plt.bar(range(len(l1)),l1.values())
plt.xticks(range(len(l1)))
plt.xlabel('No. of useful votes')
plt.ylabel('Review count')
plt.show()
#plt.xticks(range(len(Feature_importance_dict)), Feature_importance_dict.keys())
#plt.show()
###########################################
# Set up random forest regressor

#randomForest = RandomForestRegressor(n_estimators= 400, max_depth=7, random_state= 7)
#print 'Time to train randomForest regressor!'
#randomForest.fit(review_without_useful, review_train['votes_useful'])
#print 'Training complete'

#cap_result = randomForest.predict(review_valid.drop(['votes_useful'],axis = 1))
#print 'The rmsle is ' + str(rsmle(cap_result, review_valid['votes_useful']))

#############################################
# set up lasso regression
'''
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.2, max_iter=100)
print 'Train Lasso regressor!'
lasso.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete! Time for prediction!'
cap_result = lasso.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle by Lasso is: ' + str(rsmle(cap_result, review_valid['votes_useful']))
'''
##################################################
# Set up svm regression
'''
from sklearn.svm import SVR
# set up support vector regressor
supportvectorregressor = SVR(C=1.0, epsilon=0.3)
# training !
print 'Time for training!'
supportvectorregressor.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete! Time for predictions!'
cap_result = supportvectorregressor.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle by SVR  is: ' + str(rsmle(cap_result, review_valid['votes_useful']))
'''

##################################################
# Set up the bagging regressor
'''
from sklearn.ensemble import BaggingRegressor
# set up the regressor
baggingregressor = BaggingRegressor(n_estimators=1000, random_state= 7)
print 'Time for training!'
baggingregressor.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete! Time for predictions!'
cap_result = baggingregressor.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle by Bagging Regressor  is: ' + str(rsmle(cap_result, review_valid['votes_useful']))
'''

#######################################################
# Set up the adaboost regressor
'''
from sklearn.ensemble import AdaBoostRegressor
adaboost = AdaBoostRegressor(n_estimators= 1000, random_state=7)
print 'Time for training!'
adaboost.fit(review_without_useful, review_train['votes_useful'])
print 'Training complete! Time for predictions!'
cap_result = adaboost.predict(review_valid.drop(['votes_useful'], axis=1))
print 'The rmsle by Adaboost  is: ' + str(rsmle(cap_result, review_valid['votes_useful']))
'''