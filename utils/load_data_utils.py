from pandas import read_csv, concat, merge, DataFrame
import numpy as np

def load_to_df(input_filenames, output_filename, prefix = ''):
    # TODO: Take in a folder and load all the files in tha folder
    # Load in the sensor data
    dfs = []
    for filename in input_filenames:
        df = read_csv(prefix + filename)
        dfs.append(df)
    input_data = concat(dfs)
    # Standardize timestamps to 10 Hz
    # Get start and end timestamps
    start_timestamp = input_data.iloc[0]['timestamp']
    end_timestamp = input_data.iloc[-1]['timestamp']
    # Generate time steps at the same rate as accel data
    timestamp_range = np.arange(start_timestamp,end_timestamp+0.1,0.1)
    timestamp_range = np.round(timestamp_range,1)
    # Add existing data to the full df
    standardized_input = DataFrame(timestamp_range,columns=['timestamp'])
    standardized_input = merge(standardized_input,input_data, how='outer', on='timestamp')
    # Fill the data with ffil
    standardized_input.fillna(method='ffill', inplace=True)


    # Load groundtruth data
    groundtruth_data = read_csv(prefix + output_filename)
    # Drop all lights off (-1) values from groundtruth
    groundtruth_data = groundtruth_data.drop(groundtruth_data[groundtruth_data['Standing'] == -1].index)

    # Standardize timestamps to every 1 second
    # Get start and end timestamps
    # start_timestamp = groundtruth_data.iloc[0]['Unixtime']
    # end_timestamp = groundtruth_data.iloc[-1]['Unixtime']
    # # Generate time steps at the same rate as accel data
    # timestamp_range = np.arange(start_timestamp,end_timestamp+1,1)
    # timestamp_range = np.round(timestamp_range,1)
    # # Add existing data to the full df
    # standardized_output = DataFrame(timestamp_range,columns=['Unixtime'])
    # standardized_output = merge(standardized_output,groundtruth_data, how='outer', on='Unixtime')
    # # Fill the data with ffil
    # standardized_output.fillna(method='ffill', inplace=True)

    return standardized_input, groundtruth_data

def prepare_uwb_data(uwb_filenames, prefix=''):
    # Load in the uwb data (sample rate is 15 seconds)
    dfs = []
    for filename in uwb_filenames:
        df = read_csv(prefix + filename)
        dfs.append(df)
    original_uwb_df = concat(dfs)

    # Get start and end timestamps
    start_timestamp = original_uwb_df.iloc[0]['timestamp']
    end_timestamp = original_uwb_df.iloc[-1]['timestamp']
    # Generate time steps at the same rate as accel data
    filled_timestamp = np.arange(start_timestamp,end_timestamp+0.1,0.1)
    filled_timestamp = np.round(filled_timestamp,1)
    # Add existing data to the full df
    filled_uwb_df = DataFrame(filled_timestamp,columns=['timestamp'])
    filled_uwb_df = merge(filled_uwb_df,original_uwb_df, how='outer', on='timestamp')
    # Drop all rows except location and timestep
    uwb_df = filled_uwb_df.loc[:, filled_uwb_df.columns.intersection(['timestamp','location_x_m','location_y_m','location_z_m'])]
    # Fill the data with ffil
    uwb_df.fillna(method='ffill', inplace=True)
    
    return uwb_df


def create_rolling_window_data(input_df, groundtruth_df, window_size = 3, features = ['accel_x_mps2','accel_y_mps2','accel_z_mps2']):
    # Drop all columns except the features we want to train on and timestamps for data matching
    input_df = input_df.loc[:, input_df.columns.intersection(features + ['timestamp'])]

   # Get base time difference size
   # Use 3 and 2 in case there is a problem with the first index
    base_time = groundtruth_df['Unixtime'][3] - groundtruth_df['Unixtime'][2]

    # Make it a minute
    base_time = base_time * 2

    print("Base time is: " + str(base_time))
    window = base_time * window_size
    # Get grountruths in time window
    X,y = list(), list()

    for start_time in np.arange(groundtruth_df.iloc[0]['Unixtime'],groundtruth_df.iloc[-1]['Unixtime'], base_time):
        end_time = start_time + window
        # Groundtruth is given in one minute time windows, so split input data every minute
        groundtruth_data_for_time_window = groundtruth_df.loc[(groundtruth_df['Unixtime'] >= start_time) & (
            groundtruth_df['Unixtime'] < end_time)]
        labels = groundtruth_data_for_time_window['Standing'].tolist()

        # Check to make sure this isn't a transition period
        if len(set(labels)) != 1:
            continue

        # Get associated sensor data
        input_data_for_time_window = input_df.loc[(input_df['timestamp'] >= start_time) & (
            input_df['timestamp'] < end_time)]
        input_data_for_time_window = input_data_for_time_window.drop(
                'timestamp', axis=1)
        
        # Get rid of empty windows
        if len(input_data_for_time_window) == 0:
            continue

        # Make sure we have consistent shape (Standardize to 600 readings per window)
        sensor_data_list = input_data_for_time_window.values.tolist()
        expected_readings = 10*window
        if len(sensor_data_list) != expected_readings:
            continue

        # Add X data
        X.append(sensor_data_list)
        # Add y data
        y.append(labels[0])
        

    return X,y