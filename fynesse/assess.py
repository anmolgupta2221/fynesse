from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import osmnx as ox
import matplotlib.pyplot as plt

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


# Get geometries for POIs in an area around a specifed location (location described using latitude and longitude)
def get_geometries(place_name, latitude, longitude, box_size = 0.2, tag_dict = {"amenity": True, "buildings": True, "historic": True, "leisure": True, "shop": True, "tourism": True, "public transport" : True}):
  place_name = place_name
  latitude = latitude
  longitude = longitude
  north = latitude + box_size/2
  south = latitude - box_size/2
  west = longitude - box_size/2
  east = longitude + box_size/2
  return ox.geometries_from_bbox(north, south, east, west, tag_dict)

def explore_keys(pois,keys = ["name", "addr:city", "addr:postcode", "amenity", "building",
        "historic",
        "memorial",
        "religion",
        "tourism",
        "education",
        "emergency",
        "leisure",
        "shop",
        "public_transport",
        "height",
        "healthcare",
        "school"]):
    
    not_present = []
    for key in keys:
        if key not in pois.columns:
          not_present.append(key)

    if not_present:
        print(f"These keys were not present: {not_present}")
    else:
        print("All keys were present.")
    
    present_keys = [key for key in keys if key in pois.columns]
    return pois[present_keys]

def dataframe_cleanup(df):
  df = pd.DataFrame(df)
  df = df.reset_index()
  return df

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
