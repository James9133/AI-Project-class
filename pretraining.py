# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import missingno as msno
import folium
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# !pip -q install utm
import utm

import math
import folium
import pickle

from matplotlib.pyplot import xticks
from haversine import haversine
from datetime import datetime
from collections import Counter
# from imblearn.over_sampling import SMOTE
from numpy import where
import statsmodels.api as sm

from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, confusion_matrix, RocCurveDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lightgbm import LGBMClassifier

from numpy import mean
from numpy import std

from sklearn.model_selection import RepeatedStratifiedKFold

from geopy.geocoders import Nominatim

# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
# %config InlineBackend.figure_format = 'retina'


train = pd.read_csv(r'D:\AI\chonnam\courses\3rd semester\AI project\fake GPS detection\archive\train.csv')
test = pd.read_csv(r'D:\AI\chonnam\courses\3rd semester\AI project\fake GPS detection\archive\test.csv')
sample = pd.read_csv(r'D:\AI\chonnam\courses\3rd semester\AI project\fake GPS detection\archive\sample_submission.csv')


def plot_folium(df, order_id, lat_column, lon_column, location, zoom_start=10):
  # Select subset of dataframe by order ID
  df = df[df.order_id==order_id]

  # Folium plot
  my_map = folium.Map(location=location, zoom_start=zoom_start)  

  # Define different colors for status
  for index, row in df.iterrows():
    if row.driver_status=='UNAVAILABLE':
      color = 'red'
    if row.driver_status=='AVAILABLE':
      color = 'green'
    if row.driver_status=='OTW_PICKUP':
      color = 'black'
    if row.driver_status=='OTW_DROPOFF':
      color = 'blue'

    # Plot coordinates on Folium
    folium.CircleMarker([row[lat_column], row[lon_column]],
                        radius=5, color=color,
                        fill=True).add_to(my_map)
  # display(my_map)
    
# Define function for cos/sin transformations
def calculate_day_cos(row, day_attribute):
    day_cos = math.cos(2 * math.pi * row[day_attribute] / 7)
    return day_cos

def calculate_day_sin(row, day_attribute):
    day_sin = math.sin(2 * math.pi * row[day_attribute] / 7)
    return day_sin

def calculate_hour_cos(row, hour_attribute):
    hour_cos = math.cos(2 * math.pi * row[hour_attribute] / 24)
    return hour_cos

def calculate_hour_sin(row, hour_attribute):
    hour_sin = math.sin(2 * math.pi * row[hour_attribute] / 24)    
    return hour_sin
def calculate_distance(row):
    return haversine((row['latitude'], row['longitude']),(row['prev_latitude'], row['prev_longitude']), unit='km')

def gojek_data_transform_2(df):
  # Convert Linux seconds to datetime format
  df['linux_date'] = [datetime.utcfromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S') for s in df.seconds.values]

  # Convert datetime to Pandas format
  df['linux_date'] = pd.to_datetime(df['linux_date'])

  # Convert datetime in date column to Pandas format
  df['date'] = pd.to_datetime(df['date'])


  # Differencing some columns
  df['longitude_diff'] = df.groupby('order_id').longitude.diff().fillna(0)
  df['latitude_diff'] = df.groupby('order_id').latitude.diff().fillna(0)
  df['seconds_diff'] = df.groupby('order_id').seconds.diff().fillna(0)
  df['accuracy_diff'] = df.groupby('order_id').accuracy_in_meters.diff().fillna(0)
  df['altitude_diff'] = df.groupby('order_id').altitude_in_meters.diff().fillna(0)

  # Convert lat lon to UTM
  lat, lon = df.latitude.values, df.longitude.values
  x = utm.from_latlon(lat, lon)

  df['UTMX'] = x[0]
  df['UTMY'] = x[1]

  # Function to calculate distance between two points
  distance = lambda x_dif, y_dif: np.sqrt(x_dif**2 + y_dif**2)

  # Differencing UTM coordinates
  df['UTMX_diff'] = df.groupby('order_id').UTMX.diff().fillna(0)
  df['UTMY_diff'] = df.groupby('order_id').UTMY.diff().fillna(0)

  # Calculate step distance
  df['distance'] = distance(df.UTMX_diff, df.UTMY_diff)

  # Grouping by order ID to get service type and label
  if 'label' in df.columns:
      df_grouped1 = df.groupby('order_id')[['service_type', 'label']].max()
  else:
      df_grouped1 = df.groupby('order_id')[['service_type']].max()
  # Calculate time difference between available and otw pickup status
  id = list(df_grouped1.index)

  for num_id, order_id in enumerate(id):
    # Select dataframe subset w.r.t. order id
    df_id = df[df.order_id==order_id]
    try:
      # Select available status
      avail = df_id[df_id.driver_status=='AVAILABLE']

      # Select pickup status
      pickup = df_id[df_id.driver_status=='OTW_PICKUP']

      # Record the first and last seconds of available and pickup
      t_avail0 = avail.seconds.values[0]
      t_avail1 = avail.seconds.values[1]
      t_pickup0 = pickup.seconds.values[0]
      t_pickup1 = pickup.seconds.values[-1]

      # Calculate time difference of available and pickup last and first seconds
      avail_sec_diff = t_avail1 - t_avail0
      pickup_sec_diff = t_pickup1 - t_pickup0

    except:
      # Set time difference to Null of there is no available/pickup status 
      avail_sec_diff = np.nan
      pickup_sec_diff = np.nan

    # Record time difference to df_grouped1
    df_grouped1.loc[order_id, 'avail_sec_diff'] = avail_sec_diff
    df_grouped1.loc[order_id, 'pickup_sec_diff'] = pickup_sec_diff
    
  # Interquartile and range function
  iqr = lambda x: np.percentile(x, 75) - np.percentile(x, 25)
  range = lambda x: np.max(x) - np.min(x)
  if 'label' in df.columns:
      df = df[['order_id', 'service_type', 'driver_status', 'distance', 'hour',
               'accuracy_in_meters', 'accuracy_diff', 'altitude_in_meters', 
               'altitude_diff', 'longitude_diff', 'latitude_diff', 'UTMX_diff', 
               'UTMY_diff', 'seconds_diff', 'label']]
      df_grouped2 = df.iloc[:,:-1]
  else:
      df = df[['order_id', 'service_type', 'driver_status', 'distance', 'hour',
                  'accuracy_in_meters', 'accuracy_diff', 'altitude_in_meters', 
                  'altitude_diff', 'longitude_diff', 'latitude_diff', 'UTMX_diff', 
                  'UTMY_diff', 'seconds_diff']]
      df_grouped2 = df
  # Calculate summary statistics
  df_grouped2 = df_grouped2.groupby('order_id').aggregate([np.mean, np.min, np.max, np.std, iqr, range])
  # Reduce multi-index
  df_grouped2.columns = ['_'.join(col).strip() for col in df_grouped2.columns.values]
  
  # Replace column name <lambda_0> to IQR and <lambda_1> to range 
  col_groupby2 = df_grouped2.columns
  col_groupby2 = [w.replace('<lambda_0>', 'IQR') for w in col_groupby2]
  col_groupby2 = [w.replace('<lambda_1>', 'range') for w in col_groupby2]

  # Update names of columns
  df_grouped2.columns = col_groupby2

  # Get dummies of driver status
  df = pd.get_dummies(df, columns=['driver_status'])

  # Count number of PING by driver status
  df_grouped3 = df.groupby('order_id')[['driver_status_AVAILABLE', 'driver_status_OTW_DROPOFF',
                                          'driver_status_OTW_PICKUP','driver_status_UNAVAILABLE']].sum()
  df_grouped4 = df[['order_id', 'altitude_in_meters']]

  # Check for each row if altitude is Null
  df_grouped4['altitude_isnan'] = df_grouped4.altitude_in_meters.isnull()

  df_grouped4 = df_grouped4.groupby('order_id')[['altitude_isnan']].sum()

  # Merge all grouped dataframe
  df = pd.concat((df_grouped1, df_grouped2, df_grouped3, df_grouped4), axis=1)

  # Encode service_type
  service_label = {'service_type': {'GO_FOOD': 0, 'GO_RIDE': 1}}
  df = df.replace(service_label)  

  return df

def evaluate_model(y_train, y_train_predict, y_test, y_test_predict):
    print("Train Accuracy :", accuracy_score(y_train, y_train_predict))
    print("Train Confusion Matrix:")
    train_confusion = confusion_matrix(y_train, y_train_predict)
    print(train_confusion)
        
    TP = train_confusion[1,1] # true positive 
    TN = train_confusion[0,0] # true negatives
    FP = train_confusion[0,1] # false positives
    FN = train_confusion[1,0] # false negatives
    
    # Let's see the sensitivity of our logistic regression model
    print("Sensitivity: ", TP / float(TP+FN))
        
    # Let's see the specificity of our logistic regression model
    print("Specificity: ", TN / float(TN+FP))
    
    # Calculate Recall
    print("Recall", TP / float(TP+FN))

    # Calculate Precision
    print("Precision: ", TP / float(FP+TP))

    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, y_test_predict))
    print("Test Confusion Matrix:")
    test_confusion = confusion_matrix(y_test, y_test_predict)
    print(test_confusion)

    TP = test_confusion[1,1] # true positive 
    TN = test_confusion[0,0] # true negatives
    FP = test_confusion[0,1] # false positives
    FN = test_confusion[1,0] # false negatives
    
    # Let's see the sensitivity of our logistic regression model
    print("Sensitivity: ", TP / float(TP+FN))
        
    # Let's see the specificity of our logistic regression model
    print("Specificity: ", TN / float(TN+FP))

    # Calculate Recall
    print("Recall: ", TP / float(TP+FN))

    # Calculate Precision
    print("Precision: ", TP / float(FP+TP))

df = gojek_data_transform_2(train)

##### Calculate correlation with target
# ####From the bar plot, we get top 10 features with highest correlation with target, as follows:

# ####distance IQR, altitude difference IQR, UTMX UTMY difference IQR, latitude longitude difference IQR, accuracy difference IQR, min of distance, and missing altitude counts

# Correlation bar plot
# df.corr()['label'][2:].sort_values(ascending=True).plot.bar(figsize=(14,5))




# ####Machine learning - training and evaluation

# ###An XGBoost classifier model is built to classify if the GPS is true (1) or fake (0).

# Feature and target
X = df.drop(columns=['label'])
y = df.label



# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Copy X_train, X_test, y_train, y_test
X_train_original, X_test_original, y_train_original, y_test_original = X_train, X_test, y_train, y_test

data_to_predict = gojek_data_transform_2(test)

Model_name = 'LightGBM'


k_folds = KFold(n_splits = 5)
if Model_name == 'Decision Tree depth = 3':
    start_time = time.time()
    X_train_dt, y_train_dt = X_train_original, y_train_original
    X_validation_dt, y_validation_dt = X_test_original, y_test_original
    dt_depth_3 = DecisionTreeClassifier(max_depth=3)
    dt_depth_3.fit(X_train_dt, y_train_dt)
    # Predict y train & y validation
    y_train_dt_depth_3_pred = dt_depth_3.predict(X_train_dt)
    y_validation_dt_depth_3_pred = dt_depth_3.predict(X_validation_dt)
    evaluate_model(y_train_dt, y_train_dt_depth_3_pred, y_validation_dt, y_validation_dt_depth_3_pred)
    
    scores = cross_val_score(dt_depth_3, X, y, cv = k_folds)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    
    X_train_all, y_train_all = X,y
    dt_depth_3.fit(X_train_all, y_train_all)
    y_pred = dt_depth_3.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    end_time = time.time()
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'Decision Tree min_samples_split=20':
    start_time = time.time()
    X_train_dt, y_train_dt = X_train_original, y_train_original
    X_validation_dt, y_validation_dt = X_test_original, y_test_original
    dt_min_split = DecisionTreeClassifier(min_samples_split=20)
    dt_min_split.fit(X_train_dt, y_train_dt)
    # Predict y train & y validation
    y_train_dt_min_split_pred = dt_min_split.predict(X_train_dt)
    y_validation_dt_min_split_pred = dt_min_split.predict(X_validation_dt)
    evaluate_model(y_train_dt, y_train_dt_min_split_pred, y_validation_dt, y_validation_dt_min_split_pred)
    scores = cross_val_score(dt_min_split, X, y, cv = k_folds)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    
    X_train_all, y_train_all = X,y
    dt_min_split.fit(X_train_all, y_train_all)
    y_pred = dt_min_split.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    end_time = time.time()
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'Decision Tree min_samples_leaf=20':
    start_time = time.time()
    X_train_dt, y_train_dt = X_train_original, y_train_original
    X_validation_dt, y_validation_dt = X_test_original, y_test_original
    dt_min_leaf = DecisionTreeClassifier(min_samples_leaf=20)
    dt_min_leaf.fit(X_train_dt, y_train_dt)
    # Predict y train & y validation
    y_train_dt_min_leaf_pred = dt_min_leaf.predict(X_train_dt)
    y_validation_dt_min_leaf_pred = dt_min_leaf.predict(X_validation_dt)
    evaluate_model(y_train_dt, y_train_dt_min_leaf_pred, y_validation_dt, y_validation_dt_min_leaf_pred)
    scores = cross_val_score(dt_min_leaf, X, y, cv = k_folds)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    
    X_train_all, y_train_all = X,y
    dt_min_leaf.fit(X_train_all, y_train_all)
    y_pred = dt_min_leaf.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    end_time = time.time()
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'random forest':
    start_time = time.time()
    # Copy train & validation set to work with the model
    X_train_rf, y_train_rf = X_train_original, y_train_original
    X_validation_rf, y_validation_rf = X_test_original, y_test_original
    #rf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=3)
    rf = RandomForestClassifier()
    rf.fit(X_train_rf, y_train_rf)
    # Predict y train & y validation
    y_train_rf_pred = rf.predict(X_train_rf)
    y_validation_rf_pred = rf.predict(X_validation_rf)
    evaluate_model(y_train_rf, y_train_rf_pred, y_validation_rf, y_validation_rf_pred)
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(rf, X_train_rf, y_train_rf, scoring='accuracy', cv=cv, n_jobs=-1, 
                               error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    X_train_all, y_train_all = X,y
    rf.fit(X_train_all, y_train_all)
    y_pred = rf.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    end_time = time.time()
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'Random Forest with Random Search':
    start_time = time.time()
    X_train_rf, y_train_rf = X_train_original, y_train_original
    X_validation_rf, y_validation_rf = X_test_original, y_test_original
    # Number of trees in random forest
    n_estimators = np.linspace(100, 3000, int((3000-100)/200) + 1, dtype=int)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
    # Minimum number of samples required to split a node
    # min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
    min_samples_split = [1, 2, 5, 10, 15, 20, 30]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Criterion
    criterion=['gini', 'entropy']
    random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}
    rf_base = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf_base,
                                   param_distributions = random_grid,
                                   n_iter = 30, cv = 5,
                                   verbose=2,
                                   random_state=42, n_jobs = 4)
    rf_random.fit(X_train_rf, y_train_rf)
    rf_random.best_params_
    #Predict y train & y validation
    y_train_rf_random_pred = rf_random.predict(X_train_rf)
    y_validation_rf_random_pred = rf_random.predict(X_validation_rf)
    evaluate_model(y_train_rf, y_train_rf_random_pred, y_validation_rf, y_validation_rf_random_pred)
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(rf_random, X_train_rf, y_train_rf, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    X_train_all, y_train_all = X,y
    rf_random.fit(X_train_all, y_train_all)
    y_pred = rf_random.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    end_time = time.time()
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'XGBoost':
    
    # Copy train & validation set to work with the model
    X_train_xb, y_train_xb = X_train_original, y_train_original
    X_validation_xb, y_validation_xb = X_test_original, y_test_original
    #rf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=3)
    xb = XGBClassifier()
    
    xb.fit(X_train_xb, y_train_xb)
    
    # Predict y train & y validation
    y_train_xb_pred = xb.predict(X_train_xb)
    y_validation_xb_pred = xb.predict(X_validation_xb)
    evaluate_model(y_train_xb, y_train_xb_pred, y_validation_xb, y_validation_xb_pred)
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(xb, X_train_xb, y_train_xb, scoring='accuracy', cv=cv, n_jobs=-1, 
                               error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    X_train_all, y_train_all = X,y
    start_time = time.time()
    xb.fit(X_train_all, y_train_all)
    end_time = time.time()
    y_pred = xb.predict(data_to_predict)
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
elif Model_name == 'LightGBM':
    
    # Copy train & validation set to work with the model
    X_train_lgb, y_train_lgb = X_train_original, y_train_original
    X_validation_lgb, y_validation_lgb = X_test_original, y_test_original
    
    # Create and train LightGBM model
    lgb_model = LGBMClassifier()
    lgb_model.fit(X_train_lgb, y_train_lgb)
    
    # Predict on training and validation sets
    y_train_lgb_pred = lgb_model.predict(X_train_lgb)
    y_validation_lgb_pred = lgb_model.predict(X_validation_lgb)
    
    # Evaluate the model
    evaluate_model(y_train_lgb, y_train_lgb_pred, y_validation_lgb, y_validation_lgb_pred)
    
    # Perform cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(lgb_model, X_train_lgb, y_train_lgb, scoring='accuracy', cv=cv, n_jobs=-1, 
                               error_score='raise')
    
    # Report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    # Train the model on all data
    X_train_all, y_train_all = X, y
    start_time = time.time()
    lgb_model.fit(X_train_all, y_train_all)
    end_time = time.time()
    # Make predictions on data_to_predict
    y_pred = lgb_model.predict(data_to_predict)
    y_pred_proba = lgb_model.predict_proba(data_to_predict)
    # Create submission DataFrame
    submission = data_to_predict.reset_index().iloc[:,:1]
    submission['label'] = y_pred
    
    print(f"Runtime for {Model_name}: {end_time - start_time} seconds")
pickle.dump(lgb_model,open('lgb_model.pkl','wb'))
model=pickle.load(open('lgb_model.pkl','rb'))

# # Save pipeline into pickle
# import joblib
# joblib.dump(pipe, './gojek_xgboost.pkl')
# features = data_to_predict.loc[[F10]]
# # Confusion matrix of test set
# cm = confusion_matrix(y_pred, y_test) 
# # 使用 seaborn 的热图可视化混淆矩阵
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('预测标签')
# plt.ylabel('真实标签')
# plt.title('混淆矩阵')
# plt.show()

# # Classification report
# print(classification_report(y_test, y_pred))

# # Generate class membership probabilities
# y_pred_probs = pipe.predict_proba(X_test)

# classes = [0,1]

# # For each class
# plt.figure()
# for i, clas in enumerate(classes):
#   # Calculate False Positive Rate, True Negative Rate
#   fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:,i], 
#                                    pos_label = clas) 
  
#   # Calculate AUC
#   auroc = auc(fpr, tpr)
  
#   # Plot ROC AUC curve for each class
#   plt.plot(fpr, tpr, label=f'{clas}, AUC: {auroc:.2f}')
#   plt.plot([0, 1], [0, 1], 'k--')

# plt.title('ROC AUC')
# plt.xlabel('FPR'); plt.ylabel('TPR')
# plt.xlim(0,1); plt.ylim(0,1)
# plt.legend()
# plt.show()

# plt.figure()
# # Create a pd.Series of features importances
# fimp = pipe.steps[1][1].feature_importances_
# importances = pd.Series(data=fimp,
#                         index= X_train.columns)

# # Sort importances
# importances_sorted = importances.sort_values()[-15:]

# # Draw a horizontal barplot of importances_sorted
# importances_sorted.plot(kind='barh', color='red')
# plt.title('Features Importances')
# plt.show()

##### Predict on test set¶
# ####Creating a function to transform the test data by grouping by each order ID and engineer 74 new features.



# # Read test set
# test = pd.read_csv(r'D:\AI\chonnam\courses\3rd semester\AI project\fake GPS detection\archive/test.csv')

# After transformation, the size of test set is (500, 74) where 500=number of order ID and 74=number of new features.

# Transform test set to produce 74 features






