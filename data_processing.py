# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:44:27 2019

@author: Amar Shivaram
"""
import pandas as pd # data processing

import string
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import CountVectorizer

from data_analysis import top_rated_restaurants,top_trendy_restaurants,bar_plot_column,bar_plot,word_cloud,count_plot
from model_creation import train_test_splitter,smote,accuracy,logistic_regression,support_vector_machine_hyperparameter_tuning,support_vector_machine,naive_bayes,confusionMatrix,classificationReport,sentiment_intensity


def null_value(dataframe):
    print(dataframe.isnull().sum())
    
def target_column_value(review_df,business_df):
    print(review_df.stars.value_counts()) 
    print(business_df.is_open.value_counts())

def processing_for_analysis(review_df,business_df):
    
    business_rest_df = business_df.loc[(business_df['categories'] == 'Restaurants')]

    
    review_df['name'] = review_df['business_id'].map(business_rest_df.set_index('business_id')['name'])
    top_restaurants = review_df.name.value_counts().index[:20].tolist()
    review_top = review_df.loc[review_df['name'].isin(top_restaurants)]
    
    
    ###### FIND OUT THE TOP RATED RESTAURANTS
    top_rated_restaurants(review_top,business_df)
    
    ##### FIND OUT THE TOP TRENDY RESTAURANTS
    top_trendy_restaurants(review_top)
    
    ### cities with most business
    x = business_df['city'].value_counts().sort_values(ascending = False)
    x=x.iloc[0:25]
    bar_plot_column(x)
    
    ### most reviewed business
    x = business_df['name'].value_counts().sort_values(ascending = False)
    x=x.iloc[0:25]
    bar_plot_column(x)
    
    
def restaurant_open(business_df):
    cols = ['latitude', 'longitude','review_count', 'stars']
    
    business_X = business_df[cols]
    business_Y = business_df['is_open']
    
    business_X.fillna(0.0,inplace=True)
    
    ##### split data into training and testing data
    train_X, test_X, train_y, test_y = train_test_splitter(business_X,business_Y)
    
    #### due to the presence of class imbalance, we use smote to balance the data
    X_samp,y_samp = smote(train_X,train_y)
    
    X_samp = pd.DataFrame(X_samp)
    y_samp = pd.DataFrame(y_samp)
    test_X = pd.DataFrame(test_X)
    test_y = pd.DataFrame(test_y)
    
    
    ######## running logistic regression for varying parameters and check the output scores
    L = [0.0001,0.001,0.01,0.1,1,10]
    for i in L:
        pred_y = logistic_regression(X_samp,y_samp,test_X,i)
        print("For C as "+str(i))
        accuracy(test_y,pred_y)
        
    
    
    
       
    ###### we see that the accuracy scores arent that good inspite of hyperparameter tuning. 
    ###### We see that for C=0.01 the accuracy is high. We then check the confusion matrix.. 
        
    print("Logistic Regression")   
    pred_y = logistic_regression(X_samp,y_samp,test_X,0.01)
    
    accuracy(test_y,pred_y)
    
    confusionMatrix(test_y,pred_y)
    classificationReport(test_y,pred_y)
    
    
    
    
def sentiment_analysis(review_df):
    ############ due to a huge dataset we sample 20000 instances and then process. 

    review_samp = review_df.sample(n=20000)
    
    
    ### processing the text
    text_process1 = review_samp['text'].str.lower().str.cat(sep=' ')
    text_process2 = re.sub('[^A-Za-z]+', ' ', text_process1)
    
    #### we observe that the stop words are present which makes no contribution. 
    #### Hence we remove the stop words from the data
    #### We tokenise the words and then remove the stop words
    
    stop_words = list(get_stop_words('en'))         
    nltk_words = list(stopwords.words('english'))   
    stop_words.extend(nltk_words)
    
    word_tokens = word_tokenize(text_process2)
    len(word_tokens)
    
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    
    len(filtered_sentence)
    
    #### next we remove characters below a certain threshold , say 3 and numbers also
    #### concept done using list comprehension
    
    text_process3 = [word for word in filtered_sentence if len(word) > 3]
    
    cleaned_text = [word for word in text_process3 if not word.isnumeric()]   
    
    top_N = 100
    word_dist = nltk.FreqDist(cleaned_text)
    result_df = pd.DataFrame(word_dist.most_common(top_N),columns=['Word', 'Frequency'])

    bar_plot("Word","Frequency", result_df.head(7))
    
    word_cloud(cleaned_text,'black','Most Used Words')


    
    ########## analysis on reviews
    #### SENTIMENT ANALYSIS ON REVIEWS USING TEXTBLOB
    bloblist_desc = list()
    
    reviews =review_samp['text'].astype(str)
    
    for row in reviews:
        blob = TextBlob(row)
        bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])
        
        
        
    def sentimentReview(df_polarity_desc):
        if df_polarity_desc['sentiment'] > 0:
            val = "Positive Review"
        elif df_polarity_desc['sentiment'] == 0:
            val = "Neutral Review"
        else:
            val = "Negative Review"
        return val
    df_polarity_desc['Sentiment'] = df_polarity_desc.apply(sentimentReview, axis=1)
    
    
    print(df_polarity_desc.head())
    count_plot("Sentiment",df_polarity_desc)
    
    
    
    
    ######### SENTIMENT ANALYSIS WITH INTENSITY
    print("Sentiments with the intensity of the sentiments: ")
    df_polarity_desc = sentiment_intensity(df_polarity_desc,reviews)
    
    print(df_polarity_desc.head())
    
    
    
def predict_rating(review_df):
    #### predic the rating of review either good(5) or bad(1) - working with binary classifiers


    rating_class = review_df.sample(n=20000)
    X_review=rating_class['text']
    y=rating_class['stars']
    
    def text_process(review):
        nopunc=[word for word in review if word not in string.punctuation]
        nopunc=''.join(nopunc)
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    bow_transformer=CountVectorizer(analyzer=text_process).fit(X_review)
    
    X_review = bow_transformer.transform(X_review)
    
    train_X, test_X, train_y, test_y = train_test_splitter(X_review,y)
    
    #### using SVM
    
    support_vector_machine_hyperparameter_tuning(train_X,train_y)
    
    print("SVM for predicting rating")   
    
    pred_y = support_vector_machine(train_X, train_y,test_X,50,0.001,'sigmoid')
    accuracy(test_y,pred_y)
    
    confusionMatrix(test_y,pred_y)
    classificationReport(test_y,pred_y)

    
    
    ##### USING LOGISTIC REGRESSION
    print("Logistic Regression for predicting rating")
    
    L = [0.0001,0.001,0.01,0.1,1,10]
    for i in L:
        pred_y = logistic_regression(train_X,train_y,test_X,i)
        print("For C as "+str(i))
        accuracy(test_y,pred_y)
        confusionMatrix(test_y,pred_y)
        classificationReport(test_y,pred_y)
    
    
    ##### USING NAIVE BAYESIAN
    print("Naive Bayes for predicting rating") 
    
    
    pred_y = naive_bayes(train_X, train_y,test_X)
    accuracy(test_y,pred_y)
    
    confusionMatrix(test_y,pred_y)
    classificationReport(test_y,pred_y)