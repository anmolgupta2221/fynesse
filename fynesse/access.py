from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

import yaml
import urllib
import pymysql

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

 #Code for accessing credentials saved in the notebook by the user
def database_access(url, port):
    database_details = {"url": url,
                    "port": port}
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
        username = credentials["username"]
        password = credentials["password"]
        url = database_details["url"]
    return (credentials, username, password, url)

""" Create a database connection to the MariaDB database specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
"""

def create_connection(user, password, host, database, port=3306):
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    print('hello world')
