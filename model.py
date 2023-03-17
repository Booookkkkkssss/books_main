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

from xgboost import XGBClassifier

import prepare as pr 

#--------------------------------------------------------

def df_dummies(df):
    
    dummy_df = pd.get_dummies(df[['genre','sentiment']], dummy_na=False, drop_first=[True, True])
    col_list = dummy_df.columns.tolist()
    
    df = pd.concat([df, dummy_df],axis= 1)
    
    return df, col_list

#--------------------------------------------------------

def xy_splits(train, test):
    
    X_train = train.drop(columns= "successful")
    y_train = train['successful']
    X_test = test.drop(columns= "successful")
    y_test = test['successful']
    
    X_train = X_train.reset_index(drop= True)
    y_train = y_train.reset_index(drop= True)
    X_test = X_test.reset_index(drop= True)
    y_test = y_test.reset_index(drop= True)
    
    return X_train, y_train, X_test, y_test

#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------



#--------------------------------------------------------