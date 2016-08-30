__author__ = 'Harsh'
import pandas as pd
from PreprocessBuisness import business_frame_final

def PreprocessCheckins(checkin_frame_final):
    checkin_frame_final = checkin_frame_final.drop('type', axis = 1)
    
    checkin_frame_final = checkin_frame_final.fillna(0)  
    checkin_frame_final['no of checkins'] = checkin_frame_final.apply(sum, axis = 1)
    checkin_frame_final = checkin_frame_final[['no of checkins']]
    checkin_frame_final = checkin_frame_final.drop_duplicates()
    checkin_frame_final = checkin_frame_final.reindex(business_frame_final.index)
    checkin_frame_final = checkin_frame_final.fillna(0)
    return checkin_frame_final

def main():
    # Load check-in's data into python Pandas
    checkin_frame_train = pd.read_csv('..\yelp_training_set\yelp_training_set_checkin.csv', header = 0, index_col = 'business_id')
    checkin_frame_test = pd.read_csv('..\yelp_test_set\yelp_test_set_checkin.csv', header = 0, index_col = 'business_id')
    
    checkin_frame_final = checkin_frame_train.combine_first(checkin_frame_test)
    checkin_frame_final_processed = PreprocessCheckins(checkin_frame_final)

    checkin_frame_final_processed.to_csv('..\Processed Features\Checkin_features.csv')

if __name__ == '__main__':
    main()