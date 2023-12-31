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
import pandas as pd
import mlai
import mlai.plot as plot
import ipywidgets as widgets
from ipywidgets import interact, fixed
from IPython.display import display
from ukpostcodeutils import validation

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


# Get geometries for POIs in an area around a specifed location (location described using latitude and longitude)
def get_geometries(latitude, longitude, box_size = 0.2, tag_dict = {"amenity": True, "buildings": True, "historic": True, "leisure": True, "shop": True, "tourism": True, "public transport" : True}):
  latitude = latitude
  longitude = longitude
  north = latitude + box_size/2
  south = latitude - box_size/2
  west = longitude - box_size/2
  east = longitude + box_size/2
  return (ox.geometries_from_bbox(north, south, east, west, tag_dict), north, south, east, west)

#Code to explore the keys within a POIs dataframe
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

# cleanup data indexing
def dataframe_index_cleanup(df):
  df = pd.DataFrame(df)
  df = df.reset_index(drop=True)
  return df

# display all the unique values of a column
def unique_vals_col(df, col):
   return df[col].unique()

# check whether postcode is valid
def validate_postcode(pcode):
    if validation.is_valid_postcode(pcode):
        return True
    else:
        return None


# plot graph of POI
def graph_maker(north, south, east, west, place_name, df):
  graph = ox.graph_from_bbox(north, south, east, west)
  nodes, edges = ox.graph_to_gdfs(graph)
  area = ox.geocode_to_gdf(place_name)

  fig, ax = plt.subplots(figsize=plot.big_figsize)

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs
  df.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
  plt.tight_layout()
  mlai.write_figure(directory="./maps", filename=f"{place_name}-pois.svg")

# Helper function for interactive widget
def view_df(dataframe, item="all", column=None, number=6):
    if column is None:
        column = dataframe.columns[0]
    if item=="all":
        display(dataframe.head(number))
    display(dataframe[dataframe[column]==item].head(number))

# Display an interactive widget
def display_data(df, col):
  number_slider = widgets.IntSlider(min=0, max=25, step=1, value=5)
  item_select = widgets.Select(options=["all"] + (df[col].unique().tolist()))

  _ = interact(view_df, dataframe=fixed(df),
              column=fixed(col),
              item=item_select,
              number=number_slider)
  
  # Consistent street names
def capitalize_and_remove_underscores(input_string):
  if type(input_string) == str:
    result_string = input_string.replace('_', ' ').title()
    return result_string

# Clean some data columns
def labelled(data):
    data = dataframe_index_cleanup(data)
    if 'addr:postcode' in data.columns:
        data['addr:postcode'] = (data["addr:postcode"]).astype(str)
        data['addr:postcode'] = data['addr:postcode'].apply(validate_postcode)
    if 'addr:street' in data.columns:
        data['addr:street'] = data['addr:street'].apply(capitalize_and_remove_underscores)
    return data


