import os
from matplotlib import pyplot as plt
import pandas as pd
from numpy import sqrt

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

# Calculate total acceleration magnitude
# sensor_df["accel_svm_mps2"] = sqrt(sensor_df["accel_x_mps2"]**2 + sensor_df["accel_y_mps2"]**2 + (sensor_df["accel_z_mps2"])**2)

# Get every 50th row to make data less dense
sensor_df = sensor_df.iloc[::30]

sensor_df["accel_x_mps2"] = 2* ((sensor_df["accel_x_mps2"] - sensor_df["accel_x_mps2"].min()) / (sensor_df["accel_x_mps2"].max() - sensor_df["accel_x_mps2"].min())) - 1
sensor_df["accel_y_mps2"] = 2*((sensor_df["accel_y_mps2"] - sensor_df["accel_y_mps2"].min()) / (sensor_df["accel_y_mps2"].max() - sensor_df["accel_y_mps2"].min())) - 1
sensor_df["accel_z_mps2"] = 2*((sensor_df["accel_z_mps2"] - sensor_df["accel_z_mps2"].min()) / (sensor_df["accel_z_mps2"].max() - sensor_df["accel_z_mps2"].min())) - 1
# sensor_df["accel_svm_mps2"] = 2*((sensor_df["accel_svm_mps2"] - sensor_df["accel_svm_mps2"].min()) / (sensor_df["accel_svm_mps2"].max() - sensor_df["accel_svm_mps2"].min())) - 1


for direction in ["x", "y", "z"]:
    plt.figure(figsize=(20,6))
    plt.plot(groundtruth_df["Unixtime"], groundtruth_df["Labels"]-.5, label = "Groundtruth", color='red', linestyle='solid')
    plt.scatter(sensor_df["timestamp"], sensor_df[f"accel_{direction}_mps2"], label = f"Accleration in {direction}", color='blue', s=1)

    plt.legend()
    plt.title(f"Accel {direction}")
    # Graph the sensor data

    plt.xlim([sensor_df.iloc[0]['timestamp'], sensor_df.iloc[-1]['timestamp']])


    plt.savefig(f'Accel_{direction}.png')