# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf
# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import osmnx as ox

from . import access
from . import assess

import matplotlib.pyplot as plt
from ukpostcodeutils import validation
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
import yaml
from scipy.stats import t

# Helper function for applying indicator functions to features
def indicator_function(row, feature, column_name):
    if row[column_name] == feature:
        return 1
    else:
        return 0

# Indicator function applier
def indicator(feature, column_name, df):
  return np.array(df.apply(lambda row: indicator_function(row, feature, column_name), axis=1))

# Takes a geodatafram and returns a tuple for the latitude and longitude of every row calculated using the centroid
def get_centroid(gdf):
  p_latitude = []
  p_longitude = []
  for _, p in gdf.iterrows():
    p_latitude.append(p.geometry.centroid.y)
    p_longitude.append(p.geometry.centroid.x)
  return (np.array(p_latitude), np.array(p_longitude))

# Extracts the non null values from a specified column in a specified dataframe.
def feature_picker(df,column):
  if column not in df.columns:
    return pd.DataFrame.empty
  else:
    return df[df[column].notna()]

# Pick out relevant features from df 
def relevant_pois(pois):
  amenities = feature_picker(pois, "amenity")
  schools = amenities[amenities["amenity"] == 'school']
  healthcare = feature_picker(pois, "healthcare")
  leisure = feature_picker(pois, "leisure")
  public_transport = feature_picker(pois, "public_transport")
  return(amenities, schools, healthcare, leisure, public_transport)

# Fit the model based on y data and design matrix 
def fit_model(y, design):
  m_linear_basis = sm.OLS(y,design)
  results_basis = m_linear_basis.fit()
  results_basis_0 = m_linear_basis.fit_regularized(alpha=0.10,L1_wt=0.0)
  results_basis_1 = m_linear_basis.fit_regularized(alpha=0.10,L1_wt=0.3)
  results_basis_2 = m_linear_basis.fit_regularized(alpha=0.10,L1_wt=0.6)
  results_basis_3 = m_linear_basis.fit_regularized(alpha=0.10,L1_wt=1.0)
  return (results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3)

# Make predictions based on results basis and design matrix
def predict_model(design, results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3):
  y_pred = results_basis.predict(design)
  y_pred_0 = results_basis_0.predict(design)
  y_pred_1 = results_basis_1.predict(design)
  y_pred_2 = results_basis_2.predict(design)
  y_pred_3 = results_basis_3.predict(design) 
  return(y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3)

# Extract relevant dataframes from the join 
def extract_relevant_df(latitude, longitude, date, property_box_size, date_range, conn):
  all_years_properties = access.joinq(conn, latitude, longitude, property_box_size)
  input_date = datetime.strptime(date, "%B %Y")
  relevant_years_properties = all_years_properties[(all_years_properties['date_of_transfer'] >= (input_date - timedelta(days=365*date_range))) & (all_years_properties['date_of_transfer'] <= (input_date + timedelta(days=365*date_range)))]
  return relevant_years_properties 

# Setup all the features from the property_prices database using the indicator functions
def property_prices_features(df):
  detatched = indicator('D', 'property_type', df)
  semi_detatched = indicator('S', 'property_type', df)
  flat = indicator('F', 'property_type', df)
  terraced = indicator('T', 'property_type', df)
  other = indicator('O', 'property_type', df)
  new_build = indicator('Y', 'new_build_flag', df)
  tenure_type_f = indicator('F', 'tenure_type', df)
  tenure_type_l = indicator('L', 'tenure_type', df)
  return (detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l)

# Function to count matching records in pois for each row in df
def count_matching_pois(row, pois, threshold):
    lat = row['latitude']
    lon = row['longitude']
    matching_pois = pois[
        (pois['latitude'] >= lat - threshold) & (pois['latitude'] <= lat + threshold) &
        (pois['longitude'] >= lon - threshold) & (pois['longitude'] <= lon + threshold)
    ]
    return len(matching_pois)

#Function to get the centroid of each geometry and add to pois
def adjust_pois(pois):
  lat,longit = get_centroid(pois)
  pois = assess.labelled(pois)
  pois['latitude'] = lat
  pois['longitude'] = longit
  return pois

#Function to calculate R squared value of prediction
def r_squared_calc(prediction, y):
  # Calculate the mean of the dependent variable
  y_mean = np.mean(y)

  # Calculate total sum of squares
  total_sum_of_squares = np.sum((y - y_mean)**2)

  # Calculate residual sum of squares
  residual_sum_of_squares = np.sum((y - prediction)**2)

  # Calculate R-squared
  r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
  return r_squared

#Function to return closest distance to specified dataframe containing points of interest
def closest_pois_distance(row, pois):
    lat = row['latitude']
    lon = row['longitude']
    
    # Calculate Euclidean distances to all pois
    distances = cdist([(lat, lon)], pois[['latitude', 'longitude']], metric='euclidean')[0]
    
    # Find the index of the closest pois
    closest_index = np.argmin(distances)
    
    # Return the distance to the closest pois
    return distances[closest_index]

#Function to return sigmoid (kinda) of feature
def sigmoidish(x):
    return ((1 / (1 + np.exp(-0.3 * x))) - 0.5)

# Function to pick the basis vector and outputs the equivalent predictions and r squared value
def pick_basis(results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3, y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3, y):
    bases = [results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3]
    predictions = [y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3]
    highest = 0
    index = 0
    for i, current_predictions in enumerate(predictions):
        current = r_squared_calc(current_predictions, y)
        if current > highest:
            highest = current
            index = i
    return (bases[index], predictions[index], highest)

# function to evaluate how good the model fit was using r squared value, confidence intervals and percentage difference of predictions
def evaluate_prediction(results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3,  y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3,y):
  chosen_basis, predictions, r_squared_value = pick_basis(results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3, y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3,y)
  percentage_difference = np.abs(y - predictions) / np.abs(y) * 100
  average_percentage_difference = np.mean(percentage_difference)
  filtered_percentage_difference = remove_outliers_percentage_difference(percentage_difference)
  bad_indicator_count = 0
  if (filtered_percentage_difference - average_percentage_difference) > 0.2:
    bad_indicator_count += 1
  elif (filtered_percentage_difference > 0.25):
    bad_indicator_count += 1
  elif (r_squared_value < 0.4):
    bad_indicator_count += 1
  if (bad_indicator_count >= 2):
     print({f"This prediction may be poor as the r_squared_value is {r_squared_value}, the average percentage difference between the predicted prices and actual prices is {average_percentage_difference}% with outliers and {filtered_percentage_difference} without outliers"})
  print(results_basis.summary())   
  return (r_squared_value, average_percentage_difference, filtered_percentage_difference, bad_indicator_count, chosen_basis, predictions)

# function to calculate confidence intervals
def confidence_intervals(y, predictions, confidence = 0.95):
  errors = y - predictions
  n = len(errors)
  mean_error = np.mean(errors)
  standard_error = np.std(errors, ddof=1) / np.sqrt(n)
  t_value = t.ppf((1 + confidence) / 2, n - 1)
  margin_of_error = t_value * standard_error
  lower_bound = mean_error - margin_of_error
  upper_bound = mean_error + margin_of_error
  return lower_bound, upper_bound, confidence

# remove outliers and calculate average for percentage difference
def remove_outliers_percentage_difference(percentage_difference, iqr_factor=1.5):
    # Calculate IQR for percentage difference
    q1 = np.percentile(percentage_difference, 25)
    q3 = np.percentile(percentage_difference, 75)
    iqr = q3 - q1

    # Define lower and upper bounds for outlier detection
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    # Identify indices of outliers
    outliers_indices = np.where((percentage_difference < lower_bound) | (percentage_difference > upper_bound))[0]

    # Filter out outliers from percentage difference and corresponding indices in 'data' and 'y'
    filtered_percentage_difference = np.mean(np.delete(percentage_difference, outliers_indices))

    return filtered_percentage_difference

# function to determine which of two options is more common in a df column.
def more_prevalent(df, col, indicator1, indicator2):
  counts = df[col].value_counts()
  if counts[indicator1] > counts[indicator2]:
    flag = 1
  else:
    flag = 0
  return flag


# count the number of features in bounding box based on pois
def osm_feature_count(df, amenities, schools, healthcare, leisure, public_transport, threshold):
  amenity_count = np.array(df.apply(count_matching_pois, pois=amenities, threshold=threshold, axis=1)) 
  school_count = np.array(df.apply(count_matching_pois, pois=schools, threshold = threshold, axis=1))
  healthcare_count = np.array(df.apply(count_matching_pois, pois=healthcare, threshold = threshold, axis=1))
  leisure_count = np.array(df.apply(count_matching_pois, pois=leisure, threshold = threshold, axis=1))
  p_trans_count = np.array(df.apply(count_matching_pois, pois=public_transport, threshold = threshold, axis=1))
  return(amenity_count, school_count, healthcare_count, leisure_count, p_trans_count)

def features_1(latitude, longitude, date, conn, property_box_size = 0.01, date_range = 1, osm_box_size = 0.02, feature_decider = 'variation 1', threshold = 0.01):
  df = extract_relevant_df(latitude, longitude, date, property_box_size, date_range, conn)
  tenure_flag = more_prevalent(df, "tenure_type", "F", "L")
  new_build_flag = more_prevalent(df, "new_build_flag", "Y", "N")
  detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l = property_prices_features(df)
  pois, _,_,_,_ = assess.get_geometries(latitude, longitude, osm_box_size, {'amenity':True, 'healthcare':True, 'leisure':True, 'public_transport':True})
  pois = adjust_pois(pois)
  amenities, schools, healthcare, leisure, public_transport = relevant_pois(pois)
  return (latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, pois, df, tenure_flag, new_build_flag, property_box_size, date_range, osm_box_size, feature_decider, threshold)

def features_2(latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, pois, df, tenure_flag, new_build_flag, property_box_size = 0.01, date_range = 1, osm_box_size = 0.02, feature_decider = 'variation 1', threshold = 0.01):
  amenity_count, school_count, healthcare_count, leisure_count, p_trans_count = osm_feature_count(df, amenities, schools, healthcare, leisure, public_transport, threshold)
  if (feature_decider == 'count'):
    amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature = amenity_count, school_count, healthcare_count, leisure_count, p_trans_count
  elif (feature_decider == 'variation 1'):
    amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature = np.sqrt(amenity_count), sigmoidish(school_count), np.array(df.apply(closest_pois_distance, pois=pois, axis=1)), leisure_count, np.array(df.apply(count_matching_pois, pois=public_transport, threshold = threshold/2, axis=1))
  return(detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature, tenure_flag, new_build_flag, conn, df)  
  
def make_predictions(detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature, tenure_flag, new_build_flag, df):
  design = np.concatenate((detatched.reshape(-1,1), semi_detatched.reshape(-1,1), terraced.reshape(-1,1),
                           flat.reshape(-1,1), other.reshape(-1,1), new_build.reshape(-1,1), tenure_type_f.reshape(-1,1), 
                           tenure_type_l.reshape(-1,1), amenity_feature.reshape(-1,1), school_feature.reshape(-1,1), healthcare_feature.reshape(-1,1),
                           p_trans_feature.reshape(-1,1), leisure_feature.reshape(-1,1)),axis=1)
  y = np.array(df['price']).astype(float)
  y += 0.05*np.random.randn(len(df))
  results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3 = fit_model(y, design)
  y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3 = predict_model(design, results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3)
  return (results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3, y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3, y, new_build_flag, tenure_flag)

def final_prediction(latitude, longitude, date, property_type, conn, chosen_basis, tenure_flag, new_build_flag):
    detatched_acc, semi_detatched_acc, terraced_acc, flat_acc, other_acc = 0,0,0,0,0
    if property_type == 'D':
      detatched_acc = 1
    elif property_type == 'S':
      semi_detatched_acc = 1
    elif property_type == 'T':
      terraced_acc = 1
    elif property_type == 'F':
      semi_detatched_acc = 1
    else:
      other_acc = 1
    tenure_type_f_acc, tenure_type_l_acc = 0,0 
    if tenure_flag == 1:
      tenure_type_f_acc = 1
    else:
      tenure_type_l_acc = 1
    if new_build_flag == 1:
      new_build_acc = 1
    else:
      new_build_acc = 1
    dummy_df = {
    'latitude': [latitude],
    'longitude': [longitude]}
    dummy_df = pd.DataFrame(dummy_df)
    latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, pois, df, tenure_flag, new_build_flag, property_box_size, date_range, osm_box_size, feature_decider, threshold = features_1(latitude, longitude, date, conn)
    print(df)
    print(dummy_df)
    print(type(dummy_df))
    print(type(df))
    detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature, tenure_flag, new_build_flag, conn= features_2(latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, dummy_df, tenure_flag, new_build_flag, property_box_size, date_range, osm_box_size, feature_decider, threshold)
    design = np.concatenate((detatched_acc.reshape(-1,1), semi_detatched_acc.reshape(-1,1), terraced_acc.reshape(-1,1),
                           flat_acc.reshape(-1,1), other_acc.reshape(-1,1), new_build_acc.reshape(-1,1), tenure_type_f_acc.reshape(-1,1), 
                           tenure_type_l_acc.reshape(-1,1), amenity_feature.reshape(-1,1), school_feature.reshape(-1,1), healthcare_feature.reshape(-1,1),
                           p_trans_feature.reshape(-1,1), leisure_feature.reshape(-1,1)),axis=1)

    return (chosen_basis.predict(design))

def predict_price(latitude, longitude, date, property_type, conn):
    latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, pois, df, tenure_flag, new_build_flag, property_box_size, date_range, osm_box_size, feature_decider, threshold = features_1(latitude, longitude, date, conn)
    detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature, tenure_flag, new_build_flag, conn, df= features_2(latitude, longitude, date, conn, detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenities, schools, healthcare, leisure, public_transport, pois, df, tenure_flag, new_build_flag, property_box_size, date_range, osm_box_size, feature_decider, threshold)
    results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3, y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3,y, new_build, tenure_type = make_predictions(detatched, semi_detatched, flat, terraced, other, new_build, tenure_type_f, tenure_type_l, amenity_feature, school_feature, healthcare_feature, leisure_feature, p_trans_feature, tenure_flag, new_build_flag, df)
    r_squared_value, average_percentage_difference, filtered_percentage_difference, bad_indicator, chosen_basis, predictions = evaluate_prediction(results_basis, results_basis_0, results_basis_1, results_basis_2, results_basis_3,  y_pred, y_pred_0, y_pred_1, y_pred_2, y_pred_3,y)
    print(f"The r_squared_value for the linear regression was {r_squared_value}. The average_percentage_difference between each house price and the predicted house price was {average_percentage_difference} and the percentage difference after removing the outliers is {filtered_percentage_difference} ")
    final_price = final_prediction(latitude, longitude, date, property_type, conn, chosen_basis, tenure_flag, new_build_flag)
    return (final_price, average_percentage_difference)
