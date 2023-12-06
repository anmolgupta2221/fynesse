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

# Helper function for applying indicator functions to features
def indicator_function(row, feature, column_name):
    if row[column_name] == feature:
        return 1
    else:
        return 0

# Indicator function applier
def indicator(feature, column_name):
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

