import numpy as np
import pandas as pd 
import os
import itertools
# import seaborn as sns
import pickle
# sns.set(color_codes=True)

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix 

def load_dataset(dataset_path):
    not_clean = pd.read_csv(dataset_path) 
    heart_disease_data = not_clean.iloc[:,0].apply(lambda x: pd.Series(str(x).split(";")))
    heart_disease_data.columns = ["id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active",'cardio']

    column_names = ["id","age","gender","height","weight",
                "ap_hi","ap_lo","cholesterol","gluc",
                "smoke","alco","active",'cardio']
    
    for col in column_names:
        heart_disease_data[col] = pd.to_numeric(heart_disease_data[col])
        
    # Replace missing values with column median
    heart_disease_data = heart_disease_data.fillna(heart_disease_data.median())

    
    # print(heart_disease_data.head())
    heart_disease_data=heart_disease_data.drop(labels="id", axis=1)
    heart_disease_data=heart_disease_data.drop_duplicates()
    
        
    y = heart_disease_data.cardio
    X = heart_disease_data.drop("cardio", axis=1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(train_data, train_label, n_estimators = 7, learning_rate=0.5, max_depth=7):
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=learning_rate, max_delta_step=0, max_depth=max_depth,
              min_child_weight=1, missing=np.NaN, n_estimators=n_estimators, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=0)
    
    model.fit(train_data,train_label)
    
    with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "wb") as f:
        pickle.dump(model, f)


def test_model(test_data, test_label ):
    with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
  
    accuracy_score = metrics.accuracy_score(test_label, y_pred)
    precision_score = metrics.precision_score(test_label, y_pred)
    recall_score = metrics.recall_score(test_label, y_pred)
    f1_score = metrics.f1_score(test_label, y_pred)
    
    return accuracy_score, precision_score, recall_score, f1_score
    
def run( n_estimators, learning_rate, max_depth):
    file_path = "C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/dataset/cardio_train.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path)
    n_estimators, learning_rate, max_depth = 7, 0.5,7
    train_model(train_data, train_label, n_estimators, learning_rate, max_depth)
    accuracy_score, precision_score, recall_score, f1_score = test_model(test_data, test_label)
    
    return accuracy_score, precision_score, recall_score, f1_score 
     
    
if __name__== "__main__":
    run()
    
    
    
