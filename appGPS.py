from flask import Flask,request, url_for, redirect, render_template
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import folium

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

app = Flask(__name__) #Create a Flask application instance named app using the Flask class.

model=pickle.load(open('lgb_model.pkl','rb')) #load the pretrained model

test = pd.read_csv(r'D:\AI\chonnam\courses\3rd semester\AI project\fake GPS detection\archive\test.csv')

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

data_to_predict = gojek_data_transform_2(test)


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
  map_path = r'D:\AI\chonnam\courses\3rd semester\AI project\GPS-Prediction-Website\static/map.html'
  my_map.save(map_path)
    
@app.route('/') # Use the @app.route('/') decorator to define the root route. When users access the 
                # root directory of the website, the render() function is called to render the template page named "GPS_web.html".
def render():
    return render_template("GPS_web.html")  # web design template


@app.route('/predict',methods=['POST','GET'])
# Defining the prediction route: Use the @app.route('/predict',methods=['POST','GET']) decorator 
# to define a prediction route. When users submit a form, this route is triggered. It calls the predict() function to make predictions.



def predict(): #The predict() function retrieves the order ID from the web form, then calls the feature_extraction() 
    #function to extract features. Next, it uses the pretrained model to make predictions and formats the results as a string.
    
    # order_id =[x for x in request.form.values()]  # get the order_id from the web
    # order_id = order_id[0]

    order_id = request.form.get('order_id')
    print(order_id)
    # print(final)
    # features = data_to_predict.loc[[order_id]] # feature extraction 
    if order_id is None:
        return render_template('GPS_web.html', pred='No order_id provided.')
    # data_to_predict = gojek_data_transform_2(test)
    try:
        # features = data_to_predict.loc[[str(order_id)]]  # feature extraction 
        features = data_to_predict.loc[[order_id]]
    except KeyError:
        return render_template('GPS_web.html', pred='Order ID not found.')
    
    # final=[np.array(int_features)]
    
    prediction=model.predict_proba(features) # predicting with the pretrained model
    output = float('{0:.{1}f}'.format(prediction[0][1], 2))
    # Generate and save the map
    first_row = test[test['order_id'] == order_id].iloc[0]
    latitude_value = first_row['latitude']
    longitude_value = first_row['longitude']
    plot_folium(test, order_id, 'latitude', 'longitude', [latitude_value, longitude_value], zoom_start=15)
    if float(output)>0.5:
        return render_template('GPS_web.html',pred='The GPS of {} is true.\nProbability of fake GPS is {:.2f}'.format(order_id, 1-output))
    else:
        return render_template('GPS_web.html',pred='The GPS of {} is fake.\n Probability of fake GPS is {:.2f}'.format(order_id, 1-output))

@app.route('/map', methods=['POST'])
def show_map():
    order_id = request.form.get('order_id')
    if not order_id:
        return render_template('GPS_web.html', pred='No order_id provided.')

    # Render the HTML template with the map
    return render_template('GPS_web.html', map=True)

if __name__ == '__main__':
    app.run(debug=True)
