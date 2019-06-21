# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:40:58 2019

@author: Amar Shivaram
"""
import pandas as pd #### for data processing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


##### packages for text
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.downloader.download('vader_lexicon')

def train_test_splitter(dataframe_x,dataframe_y):
    train_X, test_X, train_y, test_y = train_test_split(dataframe_x,dataframe_y,test_size = 0.3, random_state = 42)
    return(train_X, test_X, train_y, test_y)
    
def smote(train_X,train_y):
    #### Due to the presence of class imbalance, we use SMOTE to balance the dataset
    sm = SMOTE(random_state=42)
    X_samp, y_samp = sm.fit_sample(train_X, train_y)

    return(X_samp,y_samp)
    
    
def logistic_regression(X_samp,y_samp,test_X,i):
    LR = LogisticRegression(C=i)
    LR.fit(X_samp,y_samp)
    pred_y = LR.predict(test_X)
    
    return(pred_y)


def support_vector_machine_hyperparameter_tuning(X_train,y_train):
    print("SVM for predicting rating")   

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [ 50, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [50, 100, 1000]},
                        {'kernel': ['linear'], 'C': [50, 100, 1000]}
                       ]
    
    scores = ['accuracy','roc_auc']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
    


def support_vector_machine(X_train, y_train,X_test,c1,gamma1,kernel1):
    sv_model = SVC(C=c1,gamma=gamma1,kernel=kernel1)
    sv_model.fit(X_train, y_train)
    pred_y = sv_model.predict(X_test)
    return pred_y

def naive_bayes(X_train, y_train,X_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    pred_y=nb.predict(X_test)
    return pred_y
    
def accuracy(test_y,pred_y):
    accuracy_sc = accuracy_score(test_y,pred_y)
    print("Accuracy is ")
    print(accuracy_sc)
    
def confusionMatrix(test_y,pred_y):
    print("Confusion Matrix is ")
    print(confusion_matrix(test_y,pred_y))
    
def classificationReport(test_y, pred_y):
    print("Classification Report is ")
    print(classification_report(test_y, pred_y))


def sentiment_intensity(df_polarity_desc,reviews):
    intensity_analyzer = SentimentIntensityAnalyzer()
    
    
    listy = []
    
    for row in reviews:
      ss = intensity_analyzer.polarity_scores(row)
      listy.append(ss)
      
    se = pd.Series(listy)
    print(se.values)
    df_polarity_desc['polarity'] = se.values

    return df_polarity_desc    
