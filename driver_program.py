# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:02:36 2019

@author: Amar Shivaram
"""
import pandas as pd # data processing
import json


from data_analysis import null_values_heatmap,count_plot
from data_processing import null_value,target_column_value,restaurant_open,processing_for_analysis,sentiment_analysis,predict_rating


def driver_function(prop):
    ##### IMPORTING THE DATAFILES
    data_directory = prop['DatasetDirectory']['filepath']
#    print(data_directory)
   
    business_df = pd.read_json(data_directory + str("yelp_academic_dataset_business.json"), lines=True)
#    print("REad over")
   
    business_df = pd.read_json("./yelp-dataset/yelp_academic_dataset_business.json", lines=True)
   #checkin_df = pd.read_json("./yelp-dataset/yelp_academic_dataset_checkin.json", lines=True)
   #tip_df = pd.read_json("./yelp-dataset/yelp_academic_dataset_tip.json", lines=True)
   #user_df = pd.read_json("./yelp-dataset/yelp_academic_dataset_user.json", lines=True)
   
   # Load Yelp reviews data
    review_file = open(data_directory + str("yelp_academic_dataset_review.json"), encoding="utf8")
    review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(business_df.shape[0])])
    review_file.close()
   
#    print("Read review over")
   
    #review_df = pd.read_csv("./yelp-dataset/yelp_academic_dataset_review.csv")
    
    #####################################################################
    
    ##### CHECK TARGET VALUE VALUE COUNT
    target_column_value(review_df,business_df)


    ##### CHECK THE NULL VALUES PRESENT
    null_value(review_df)
    null_value(business_df)
    
    ##### SELECTING THE REVIEWS WHICH ARE 1 OR 5 SO THAT LATER WE CAN PERFORM A USECASE ON THIS    
    review_df = review_df[(review_df['stars'] == 1) | (review_df['stars'] == 5)]

    processing_for_analysis(review_df,business_df)
    
    
    ##### PLOT THE HEATMAP OF NULL VALUES FOR BOTH DATAFRAMES
#    print("Heatmap for business data")
    null_values_heatmap(business_df)
#    print("Heatmap for reviews data")
    null_values_heatmap(review_df)

    ##### COUNT PLOT FOR THE TARGET COLUMN OF THE PROBLEMS UNDER CONSIDERATOIN
    count_plot('is_open',business_df)
    

    #### process and find the top rated and top trendy restaurants
    processing_for_analysis(review_df,business_df)
    
    
    #### usecase 1 - to check if the restaurant is open or not
    restaurant_open(business_df)
    
    
    ### usecase -2 -to process reviews and carry out sentiment analysis
    sentiment_analysis(review_df)
    
    
    ### usecase 3 - to predict the review rating from the review text available
    predict_rating(review_df)