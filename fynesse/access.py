from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

import yaml

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

# Code for requesting and storing credentials (username, password) of the RDS

def database_access(url, port):
    database_details = {"url": url,
                    "port": port}
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
        username = credentials["username"]
        password = credentials["password"]
        url = database_details["url"]
    return credentials,username,password,url


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    print('hello world')
