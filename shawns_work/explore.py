import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.model_selection

#-------------------------------------------------------

def split(df):
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.Target)
    return train, test