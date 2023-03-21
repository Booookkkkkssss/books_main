from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics as m

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, confusion_matrix, plot_confusion_matrix, f1_score, ConfusionMatrixDisplay

import xgboost as xgb 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import prepare as pr 

seed = 42
#--------------------------------------------------------

def ready_df(df):
    
    '''
    Input: Dataframe
    Output: A Dataframe that is ready for modeling work
    
    This function will drop key columnns from a dataframe and make dummies for key features.
    Also performs a concat of the dummies and adds back to the new dataframe.
    
    '''
    
    # drop columns we do not want to model on
    df = df.drop(columns= ['title','summary','year_published','author','reviews','cleaned_title','cleaned_summary'])
    
    # creating dummies for genre and sentient
    dummy_df = pd.get_dummies(df[['genre','sentiment']], dummy_na=False, drop_first=[True, True])
    
    # add dummies to dataframe 
    df = pd.concat([df, dummy_df],axis= 1)
    
    return df

#--------------------------------------------------------

def Xy_set(train,test):
    
    
    '''
    Input: A train and test dataframe.
    Output: X_train, y_train, X_test, y_test dataframe, where the X are features that we are modeling and y is the target feature.
    
    This function creates dataframes where the x and y are split into train and test for modeling. It will return x_train, y_train, x_test y_test where the index is reset and NOT reordered.
    '''
    # creating train and test (x and y) subsets
    X_train = train.drop(columns= "successful")
    y_train = train['successful']
    
    # creating train and test (x and y) subsets
    X_test = test.drop(columns= "successful")
    y_test = test['successful']
    
    # reset index, (no sorting or reordering)
    X_train = X_train.reset_index(drop= True)
    y_train = y_train.reset_index(drop= True)
    X_test = X_test.reset_index(drop= True)
    y_test = y_test.reset_index(drop= True)
    
    return X_train, y_train, X_test, y_test

#--------------------------------------------------------

def scaling(X_train, X_test):
    
    '''
    Input: A X_train and X_test dataframe.
    Output: X_train_scaled, X_test_scaled dataframe, where the train and test are scaled.
    
    This function takes X_train and X_test and creates a scaler object with X_train(fit). It then transforms specific numeric columns on the dataframes, adds ['neg','neutral','pos','compound'] columns from their respective X_train and X_test.
    
    '''
    
    
    # create a subset of numerical column
    xtrainnums = X_train[['review_count','number_of_ratings','length','rating']]
    
    number_list = ['review_count','number_of_ratings','length','rating']

    # Note that we only call .fit with the training data
    scaler = sklearn.preprocessing.StandardScaler()
    
    # fit training data to scaler, not transforming here
    scaler.fit(xtrainnums)
    
    # transform the numerical values that we want based on the trained fit scaler
    X_train_scaled = scaler.transform(X_train[number_list])
    X_test_scaled = scaler.transform(X_test[number_list])
    
    # create a dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns= [number_list])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns= [number_list])
    
    
    # add the 'neg','neutral','pos','compound' from x_train to the scaled data. reset
    X_train_scaled[['neg','neutral','pos','compound']] = X_train[['neg','neutral','pos','compound']].reset_index(drop = True)
    X_test_scaled[['neg','neutral','pos','compound']] = X_test[['neg','neutral','pos','compound']].reset_index(drop = True)

    # create a list of the dummies 
    dummies = X_train.columns.tolist()[11:]
    
    # add dummies to dataframe
    X_train_scaled = pd.concat([X_train_scaled, X_train[dummies]],axis = 1 )
    X_test_scaled = pd.concat([X_test_scaled, X_test[dummies]],axis = 1 )
    
    return X_train_scaled, X_test_scaled

#--------------------------------------------------------

def XGBclf(X_train_scaled, X_test_scaled, y_train, y_test): 
    
    ''' 
    Input: X_train_scaled, X_test_scaled dataframe.
    Output: A list of predictions from the test set, and prints out a confusion matrix to see how are model is performing.
    
    # An XGBClassifier will be created using certain ( predetermined )parameters. A list is created define the top features that help our model succeed. The model is then fit with train, and predictions are made from the test set.
    
    '''
    
    
    # create  an instance with predetermined values, using cross validation, grid search and other methods,
    # these parameters have been predetermined for the top performance. 
    xgb_clf = xgb.XGBClassifier(objective ='binary:logistic', 
                                        seed = 42,
                                        max_depth = 3,    
                                        scale_pos_weight= 7,
                                        learning_rate = .1,
                                        subsample = .7,
                                        colsample_bytree = .7,
                                        n_jobs = 10)
    
    most_imp = [('number_of_ratings',),
                'genre_Mystery',
                ('review_count',),
                 'genre_Nonfiction',
                'genre_Horror',
                 ('length',),
                 'genre_Fiction',
                 ('rating',),
                 'sentiment_very negative',
                 'genre_Young Adult',
                 'genre_Fantasy',
                 'genre_Romance',
                 ('neutral',),
                 ('neg',),
                 ('pos',),
                 ('compound',),
                 'sentiment_very positive',
                 'genre_Thriller']
    
    # fit the model with x_train, using the most important features, and y_train
    xgb_clf.fit(X_train_scaled[most_imp],y_train)
    
    # y predictions for test
    y_pred = xgb_clf.predict(X_test_scaled[most_imp])
    
    # assume y_test and y_pred are your test set target variable and predicted labels, respectively
    cm = confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Unsuccessful', 'Bestseller'])
    disp.plot()
    disp.ax_.set_title("Confusion Matrix for XGB Classifier")

    plt.show()
    
    return y_pred


#--------------------------------------------------------

def roc(y_test, y_pred):
    
    y_test = pd.DataFrame(y_test)
    
    mode_val = y_test['successful'].mode()[0]

    # Create a new column with the mode value
    y_test = y_test.assign(baseline=mode_val)
    
    
    plt.figure(figsize=(10,6))


    fpr, tpr, thresholds = roc_curve(y_test['successful'], y_pred)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'XGBClassifer (area = %0.4f)' % auc(fpr, tpr))

    fpr, tpr, thresholds = roc_curve(y_test['successful'], y_test['baseline'])
    plt.plot(fpr, tpr, color='red', lw=2, label=f'Baseline (area = %0.4f)' % auc(fpr, tpr))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('XGB Classifier Captured Area', fontsize=17)
    plt.legend(loc='lower right', fontsize=13)
    plt.show()


#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------