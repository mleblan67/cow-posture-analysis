import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from numpy import sqrt
import numpy as np
import datetime

from utils.load_data_utils import load_to_df



tag = 7
prefix = 'converted_data/'


# Build folder path
data_dir = prefix + 'T' + str(tag).zfill(2) + '/'

# Get all sensor data files for this folder
filepaths = os.listdir(data_dir)
filepaths = [data_dir + file for file in filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
filepaths.sort() # Make sure they're in order for processing

# Get groundtruth path
groundtruth_path = data_dir + 'T' + str(tag).zfill(2) + '_groundtruths.csv'

# Load in the data
sensor_df, groundtruth_df = load_to_df(filepaths, groundtruth_path)

# groundtruth_df = groundtruth_df.loc[(groundtruth_df['Unixtime'] >= sensor_df.iloc[0]['timestamp']) & (
#             groundtruth_df['Unixtime'] < sensor_df.iloc[-1]['timestamp'])]

# Calculate total acceleration magnitude
# sensor_df["accel_svm_mps2"] = sqrt(sensor_df["accel_x_mps2"]**2 + sensor_df["accel_y_mps2"]**2 + (sensor_df["accel_z_mps2"])**2)

# Get every 50th row to make data less dense
sensor_df = sensor_df.iloc[::30]

sensor_df["accel_x_mps2"] = 2* ((sensor_df["accel_x_mps2"] - sensor_df["accel_x_mps2"].min()) / (sensor_df["accel_x_mps2"].max() - sensor_df["accel_x_mps2"].min())) - 1
sensor_df["accel_y_mps2"] = 2*((sensor_df["accel_y_mps2"] - sensor_df["accel_y_mps2"].min()) / (sensor_df["accel_y_mps2"].max() - sensor_df["accel_y_mps2"].min())) - 1
sensor_df["accel_z_mps2"] = 2*((sensor_df["accel_z_mps2"] - sensor_df["accel_z_mps2"].min()) / (sensor_df["accel_z_mps2"].max() - sensor_df["accel_z_mps2"].min())) - 1
# sensor_df["accel_svm_mps2"] = 2*((sensor_df["accel_svm_mps2"] - sensor_df["accel_svm_mps2"].min()) / (sensor_df["accel_svm_mps2"].max() - sensor_df["accel_svm_mps2"].min())) - 1

# Convert to datetimes

groundtruth_df['date'] = pd.to_datetime(groundtruth_df['Unixtime'], unit='s')
sensor_df['date'] = pd.to_datetime(sensor_df['timestamp'], unit='s')

for direction in ["x", "y", "z"]:
    plt.figure(figsize=(20,6))
    ax = plt.plot(groundtruth_df["date"], groundtruth_df["Labels"]-.5, label = "Groundtruth", color='red', linestyle='solid')
    plt.scatter(sensor_df["date"], sensor_df[f"accel_{direction}_mps2"], label = f"Accleration in {direction}", color='blue', s=1)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate() 

    plt.legend()
    plt.title(f"Accel {direction}")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    # Graph the sensor data

    plt.xlim([sensor_df.iloc[0]['date'], sensor_df.iloc[-1]['date']])

    plt.savefig(f'Accel_{direction}.png')