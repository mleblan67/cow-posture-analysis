import os
from matplotlib import pyplot as plt
import pandas as pd

from utils.load_data_utils import load_to_df



tag = 3
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


# Graph the sensor data
ax = sensor_df.plot('timestamp','accel_y_mps2', kind='scatter')
groundtruth_df.plot('Unixtime', 'Labels', kind='scatter', ax=ax, color='red')

plt.show()