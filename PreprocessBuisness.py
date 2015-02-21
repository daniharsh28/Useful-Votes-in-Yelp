__author__ = 'Harsh'
import pandas as pd
import numpy as np
import operator
from sklearn.feature_extraction import DictVectorizer
#########################################################################################
#### Load the business from python pandas
business_frame_train = pd.read_csv('E:\Fall 2014\Social Media Mining\yelp_training_set\yelp_training_set_business.csv', header = 0, index_col='business_id')
business_frame_test = pd.read_csv('E:\Fall 2014\Social Media Mining\yelp_test_set\yelp_test_set_business.csv', header = 0, index_col = 'business_id')

business_frame_final = business_frame_train.combine_first(business_frame_test)
business_frame_final = business_frame_final.drop_duplicates()


#############################################################################################
def processBusiness(business_frame):
    columns_to_drop = ['full_address','latitude','longitude','name','neighborhoods','state','type']
    business_frame = business_frame.drop(columns_to_drop, axis=1)

    #####################
    #Identify the categories in each business

    category_list = business_frame.categories.fillna("").map(lambda x: str.split(x, ",")).values

    #Identifying frequent categories by checking how many times it occurs in each review
    ##We can take first 20 or 40 such categories as our features
    frequent_categories = pd.Series([category for categories in category_list for category in categories]).value_counts().ix[1:40].index
    print frequent_categories
    dictionary_vector = DictVectorizer()
    def category_to_dictionary(cats):
        final_dict = {}
        for x in cats:
            if x in frequent_categories:
                final_dict.setdefault(x, 1)
            return final_dict
    category_features = dictionary_vector.fit_transform(business_frame.categories.fillna("").map(lambda x: category_to_dictionary(str.split(x, ","))).values).toarray()
    print category_features
    category_frame = pd.DataFrame(category_features, index = business_frame.index, columns= dictionary_vector.feature_names_)

    #############
    # Combining the extracted features
    business_frame = business_frame.combine_first(category_frame)

    #####################
    # Finding the which city occurs more time by finding the value counts of it

    city_frequency = business_frame.city.value_counts()
    city_frequency_dictionary = dict(city_frequency)
    ##############################################################
    # finding the most frequent city by identifying the cities that have count > 200

    frequent_city_200 =  [(k,v) for k,v in city_frequency_dictionary.items() if v > 200]
    frequent_city_200 = dict(frequent_city_200)
    sorted_frequency_city_200 = sorted(frequent_city_200.items(), key= operator.itemgetter(1), reverse= True)
    sorted_frequency_city_200 = [str(city[0]) for city in sorted_frequency_city_200]
    print sorted_frequency_city_200
    # Sorted cities based on the count
    business_frame['city'] = business_frame.city.map(lambda test: test if test in sorted_frequency_city_200 else 'Other' )
    business_frame = business_frame.drop(['categories'], axis= 1)
    return business_frame
###############################################################
business_frame_processed = processBusiness(business_frame_final)
print business_frame_processed
#################################################################
# Save the returned final business frame into csv
business_frame_processed.to_csv('E:\Fall 2014\Social Media Mining\Processed Features\Business_features.csv')