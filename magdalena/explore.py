import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.model_selection

#-------------------------------------------------------

def split(df):
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.Target)
    return train, test








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

