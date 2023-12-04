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

# Access credentials saved in the notebook by the user
def database_access(url, port):
    database_details = {"url": url,
                    "port": port}
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
        username = credentials["username"]
        password = credentials["password"]
        url = database_details["url"]
    return (credentials, username, password, url)



# Create a database connection to the MariaDB database specified by the host url and database name.
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

 
 # Query n first rows of a databse table
def select_top(conn, table,  n):
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table} LIMIT {n}')

    rows = cur.fetchall()
    return rows

# Download and load data into the database helper function
def download_and_load_data_years(year, part, conn):
    url = f"http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-part{part}.csv"
    filename = f"pp-{year}-part{part}.csv"

    # Download the data file
    urllib.request.urlretrieve(url, filename)

    # Connect to the database
    cur = conn.cursor()

    # Load data into the database
    load_data_query = f"""
    LOAD DATA LOCAL INFILE '{filename}' INTO TABLE `pp_data`
    FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
    LINES STARTING BY '' TERMINATED BY '\\n';
    """
    cur.execute(load_data_query)


# Download and load data into the databse table
def download_data_task_a(years, parts_per_year, conn):
  # Loop through the years and parts to download and load the data

  for year in years:
      for part in range(1, parts_per_year + 1):
          download_and_load_data_years(year, part, conn)

# Count the number of rows in input table
def count_rows(conn, table):
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM {table}')

    count = cur.fetchone()[0]
    return count

# Query n first rows of the table
def select_top(conn, table,  n):
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {table} LIMIT {n}')

    rows = cur.fetchall()
    return rows

# Nice way to view the rows
def head(conn, table, n=5):
  rows = select_top(conn, table, n)
  for r in rows:
      print(r)

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    print('hello world')
