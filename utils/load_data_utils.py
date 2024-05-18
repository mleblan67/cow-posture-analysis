from pandas import read_csv, concat, merge, DataFrame, to_datetime
import numpy as np
from math import ceil

def load_to_df(input_filenames, output_filename, prefix = ''):
    # Load in the sensor data
    dfs = []
    for filename in input_filenames:
        df = read_csv(prefix + filename)
        dfs.append(df)
        
    input_data = concat(dfs)

    '''
    STANDARDIZE INPUT DATA
    Sometimes some timestamps are missing so we want to fill in missing 
    timestamps to make sure maintain a standard number of samples per minute
    '''
    # Get start and end timestamps
    start_timestamp = input_data.iloc[0]['timestamp']
    end_timestamp = input_data.iloc[-1]['timestamp']
    # Generate time steps at the same rate as the input data
    delta = input_data.iloc[1]['timestamp'] - input_data.iloc[0]['timestamp']
    # delta = 0.1 # Make delta standard in order to upsample UWB

    timestamp_range = np.arange(start_timestamp,end_timestamp+delta,delta)
    timestamp_range = np.round(timestamp_range,1)
    # Add existing data to the full df
    standardized_input = DataFrame(timestamp_range,columns=['timestamp'])
    standardized_input = merge(standardized_input,input_data, how='outer', on='timestamp')
    # Fill the data with ffil
    standardized_input.fillna(method='ffill', inplace=True)

    # Load groundtruth data
    groundtruth_data = read_csv(prefix + output_filename)

    # Convert timestamps to unixtime
    # groundtruth_data['dt'] = to_datetime(groundtruth_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    # groundtruth_data['Unixtime'] = groundtruth_data['dt'].astype(int)
    # groundtruth_data['Unixtime'] = groundtruth_data['Unixtime'].div(10**9)

    # # Fix the offset
    # # Offset is 5 hours (GMT to Chicago time over the summer)
    # offset = 3600*5
    # groundtruth_data['Unixtime'] = groundtruth_data['Unixtime'] + offset

    groundtruth_data.rename(columns={"timestamp":"Unixtime"}, inplace=True)
    return standardized_input, groundtruth_data


'''
window size is given in minutes
stride is given in minutes

After hyperparameter tuning, defualt sizes are the best tested
'''
def create_rolling_window_data(input_df, groundtruth_df, window_size = 30, stride = 5):

    # Get base time difference size
    # Use 3 and 2 in case there is a problem with the first index
    groundtruth_base_time = groundtruth_df['Unixtime'][4] - groundtruth_df['Unixtime'][3]
    print("Base time is: " + str(groundtruth_base_time))
    input_base_time = input_df['timestamp'][4] - input_df['timestamp'][3]

    # Make the base time 5 minutes to make processing much faster
    stride_time = groundtruth_base_time * stride
    window = groundtruth_base_time * window_size
    # Get grountruths in time window
    X,y = [], []

    for start_time in np.arange(groundtruth_df.iloc[0]['Unixtime'],groundtruth_df.iloc[-1]['Unixtime'], stride_time):
        end_time = start_time + window
        # Groundtruth is given in one minute time windows, so split input data every minute
        groundtruth_data_for_time_window = groundtruth_df.loc[(groundtruth_df['Unixtime'] >= start_time) & (
            groundtruth_df['Unixtime'] < end_time)]
        
        # Check to make sure this isn't a transition period
        # also get rid of unknown behavior (0)
        a = groundtruth_data_for_time_window["behavior"].to_list() # df.values
        if not (len(set(a)) == 1) or a[0] == 0:
            continue

        # Get associated sensor data
        input_data_for_time_window = input_df.loc[(input_df['timestamp'] >= start_time) & (
            input_df['timestamp'] < end_time)]
        
        # Drop columns that we don't want to use
        unused_features = ['timestamp','mag_x_uT','mag_y_uT','mag_z_uT','pressure_Pa','elevation'] #,'coord_x_cm', 'coord_y_cm']
        unused_cols = input_data_for_time_window.columns.intersection(unused_features)
        input_data_for_time_window = input_data_for_time_window.drop(columns=unused_cols)
        
        # Get rid of empty windows
        if len(input_data_for_time_window) == 0:
            continue

        # Make sure we have consistent shape (Standardize to 600 readings per window)
        sensor_data_list = input_data_for_time_window
        expected_readings = int(ceil(window/input_base_time))

        if len(sensor_data_list) != expected_readings:
            continue
        
        # Add X data
        X.append(sensor_data_list)

        # Add y data
        y.append(a[0])
        

    return np.array(X),np.array(y)