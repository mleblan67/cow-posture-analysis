import os
from pandas import read_csv, concat, merge
from numpy import sqrt
import matplotlib.pyplot as plt

prefix = "location_data/"

tags = [1,2,3,4,5,6,7,8,9,10]

# Load in all locations
location_dfs = []

# Load in all the training
for tag in tags:
    # Build folder path
    data_dir = prefix + 'T' + str(tag).zfill(2) + '/'

    # Get all UWB Location data files for this folder
    filepaths = os.listdir(data_dir)
    filepaths = [data_dir + file for file in filepaths]
    filepaths.sort() # Make sure they're in order for processing
    

    # Load in all the days for this tag
    dfs = []
    for filepath in filepaths:
        df = read_csv(prefix + filepath)
        dfs.append(df)
    # Add this tags TOTAL data to df list
    location_dfs.append(concat(dfs))


# The tag we want to see all other tags interaction with
main_tag = 4
# Get the DataFrame for our main tag
main_tag_df = location_dfs[main_tag-1]
# Rename column names for processing
main_tag_df.rename(columns={'coord_x_cm': 'x', 'coord_y_cm': 'y', 'coord_z_cm': 'z'}, inplace=True)

# Graph the distance from main_tag to all other cows over all days
for tag,df in zip(tags,location_dfs):
    # Make sure we're not comparing main_tag against itself
    if tag == main_tag:
        continue
    
    # Rename column names for processing
    df.rename(columns={'coord_x_cm': 'x1', 'coord_y_cm': 'y1', 'coord_z_cm': 'z1'}, inplace=True)
    # Merge dataframes so we can vectorize distance calculations
    distance_df = merge(main_tag_df, df, how='inner', on='timestamp')

    # TODO: Get rid of NaaN values

    # Calculate
    distance_df["distance"] = sqrt((distance_df["x"]-distance_df["x1"])^2 + (distance_df["y"]-distance_df["y1"])^2)
