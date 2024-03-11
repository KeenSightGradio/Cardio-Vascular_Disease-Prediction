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

def load_dataset(dataset_path, test_size):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(train_data, train_label, n_estimators, gamma, learning_rate, max_depth):
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=gamma,
              learning_rate=learning_rate, max_delta_step=0, max_depth=max_depth,
              min_child_weight=1, missing=np.NaN, n_estimators=n_estimators, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=0)
    
    model.fit(train_data,train_label)
    
    with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "wb") as f:
        pickle.dump(model, f)


def test_model(test_data, test_label):
    with open("C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/models/cvd_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    # Predict probabilities instead of labels
    y_pred_proba = loaded_model.predict_proba(test_data)[:, 1]
  
    accuracy_score = metrics.accuracy_score(test_label, y_pred)
    precision_score = metrics.precision_score(test_label, y_pred)
    recall_score = metrics.recall_score(test_label, y_pred)
    f1_score = metrics.f1_score(test_label, y_pred)
    
    return y_pred_proba, accuracy_score, precision_score, recall_score, f1_score
def plot_roc_curve(test_data, test_label):
    # Get predicted probabilities
    y_pred_proba, _, _, _, _ = test_model(test_data, test_label)
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image
def plot_learning_curve(train_data, train_label, test_data, test_label):
    # Load the trained model
    with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/cdr_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)

    train_errors = []
    test_errors = []
    train_sizes = [0.125, 0.25,0.5]

    for train_size in train_sizes:
        # Split the training data into smaller training set and validation set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, train_size=train_size, random_state=42)

        # Train the model on the smaller training set
        loaded_model.fit(X_train, y_train)

        # Predict on the training set and calculate the training error
        y_train_pred = loaded_model.predict(X_train)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_pred))

        # Predict on the validation set and calculate the testing error
        y_val_pred = loaded_model.predict(X_val)
        test_errors.append(metrics.mean_squared_error(y_val, y_val_pred))
     # Plot the learning curve
    plt.figure()
    plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_errors, 'o-', color="g", label="Validation error")
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.legend(loc="best")

    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    learning_curve_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    learning_curve_image = learning_curve_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return learning_curve_image

def run( n_estimators,gamma, learning_rate, max_depth, test_size):
    file_path = "C:/Users/hp/GradioApps/Cardio-Vascular-Disease-Prediction/dataset/cardio_train.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path, test_size)
    train_model(train_data, train_label, n_estimators,gamma, learning_rate, max_depth)
    _, accuracy_score, precision_score, recall_score, f1_score = test_model(test_data, test_label)
    
    
    roc_curve_image = plot_roc_curve(test_data, test_label)
    learning_curve = plot_learning_curve(train_data, train_label, test_data, test_label)
    visuals = [roc_curve_image, learning_curve]
    
    return accuracy_score, precision_score, recall_score, f1_score, visuals
     
    
if __name__== "__main__":
    run()
    
    
    
