# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:02:37 2019

@author: Amar Shivaram
"""
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def top_rated_restaurants(review_top,business_df):

    ###### TOP RATED RESTAURANTS ON YELP
    
    review_top.groupby(review_top.name)['stars'].mean().sort_values(ascending=True).plot(kind='barh',figsize=(12, 10))
    plt.yticks(fontsize=18)
    plt.title('Top rated restaurants on Yelp',fontsize=20)
    plt.ylabel('Restaurants names', fontsize=18)
    plt.xlabel('Ratings', fontsize=18)
    plt.show()


def top_trendy_restaurants(review_top):
    
    ########## top useful funny and cool restaurants
    review_top.groupby(review_top.name)[['useful','funny', 'cool']].mean().sort_values('useful',ascending=True).plot(kind='barh', figsize=(15, 14),width=0.7)
    plt.yticks(fontsize=18)
    plt.title('Top useful, funny and cool restaurants',fontsize=28)
    plt.ylabel('Restaurants names', fontsize=18)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.show()
    
def null_values_heatmap(dataframe):
    
    plt.figure(figsize=(12,10))
    sns.heatmap(dataframe.isnull(),yticklabels=False, cbar=False, cmap = 'viridis')

def count_plot(target_column,dataframe):
    sns.countplot(x=target_column,data=dataframe)
    
def bar_plot(x_value,y_value,dataframe):
    plt.figure(figsize=(6,6))
    ax= sns.barplot(x = x_value, y=y_value,data= dataframe ,alpha=0.8 )

    plt.xlabel(x_value, fontsize=12)
    plt.ylabel(y_value, fontsize=12)

def bar_plot_column(x):
    plt.figure(figsize=(16,4))
    ax = sns.barplot(x.index, x.values, alpha=0.8)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.show()
    
    
    

#### word cloud to analyse the words occuring most often
def word_cloud(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')