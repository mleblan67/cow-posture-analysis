from numpy import sqrt, arctan, arctan2, pi
from pandas import merge

def add_mean_xyz_feature(input_df, time_window_size=600):
    # Add a roling window mean of x, y, and z features using
    # size time_window_size to the input dataframe
    input_df['mean_accel_x'] = input_df.rolling(window = time_window_size, on = "timestamp").accel_x_mps2.mean()
    input_df['mean_accel_y'] = input_df.rolling(window = time_window_size, on = "timestamp").accel_y_mps2.mean()
    input_df['mean_accel_z'] = input_df.rolling(window = time_window_size, on = "timestamp").accel_z_mps2.mean()

    return input_df

def add_svm_feature(input_df):
    # Adding Signal Vector Magnitude (indicates degree of movement intensity)
    input_df['svm'] = mag(input_df['accel_x_mps2'], input_df['accel_y_mps2'], input_df['accel_z_mps2'])
    return input_df

def add_sma_feature(input_df):
    # Adding Signal Magnitude Area (distinguishes between periods of activity and rest)
    input_df['sma'] = input_df['accel_x_mps2'].abs() + input_df['accel_y_mps2'].abs() + input_df['accel_z_mps2'].abs()
    return input_df

def add_pitch_roll_features(input_df):
    input_df['pitch'] = pitch(input_df['accel_x_mps2'], input_df['accel_y_mps2'], input_df['accel_z_mps2'])
    input_df['roll'] = roll(input_df['accel_x_mps2'], input_df['accel_y_mps2'], input_df['accel_z_mps2'])
    input_df['inclination'] = inclination(input_df['accel_x_mps2'], input_df['accel_y_mps2'], input_df['accel_z_mps2'])

    return input_df

def add_uwb_features(input_df, uwb_df):
    input_df = merge(input_df,uwb_df,how='inner',on='timestamp')
    
    return input_df


def mag(x,y,z):
    return sqrt(x**2 + y**2 + z**2)

def roll(x,y,z):
    return arctan2(y,z) * (180/pi)

def pitch(x,y,z):
    return arctan(-x/(sqrt(y**2 + z**2))) * (180/pi)

def inclination(x,y,z):
    return arctan(sqrt(x**2 + y**2)/z) * (180/pi)